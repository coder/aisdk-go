package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/ssestream"
)

// ToolsToOpenAI converts the tool format to OpenAI's API format.
func ToolsToOpenAI(tools []Tool) []openai.ChatCompletionToolParam {
	openaiTools := make([]openai.ChatCompletionToolParam, len(tools))
	for _, tool := range tools {
		var schemaParams map[string]any
		if tool.Schema.Properties != nil {
			schemaParams = map[string]any{
				"type":       "object",
				"properties": tool.Schema.Properties,
			}
			if len(tool.Schema.Required) > 0 {
				schemaParams["required"] = tool.Schema.Required
			}
		}
		openaiTools = append(openaiTools, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: param.NewOpt[string](tool.Description),
				Parameters:  schemaParams,
			},
		})
	}
	return openaiTools
}

// MessagesToOpenAI converts internal message format to OpenAI's API format.
func MessagesToOpenAI(messages []Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	openaiMessages := []openai.ChatCompletionMessageParamUnion{}

	for _, message := range messages {
		switch message.Role {
		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(message.Content))
		case "user":
			content := []openai.ChatCompletionContentPartUnionParam{}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content = append(content, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: part.Text,
						},
					})
				case PartTypeFile:
					content = append(content, openai.ChatCompletionContentPartUnionParam{
						OfImageURL: &openai.ChatCompletionContentPartImageParam{
							ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
								URL: fmt.Sprintf("data:%s;base64,%s", part.MimeType, base64.StdEncoding.EncodeToString(part.Data)),
							},
						},
					})
				}
			}

			for _, attachment := range message.Attachments {
				content = append(content, openai.ChatCompletionContentPartUnionParam{
					OfImageURL: &openai.ChatCompletionContentPartImageParam{
						ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
							URL: attachment.URL,
						},
					},
				})
			}

			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfArrayOfContentParts: content,
					},
				},
			})
		case "assistant":
			content := &openai.ChatCompletionAssistantMessageParam{}

			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content.Content.OfArrayOfContentParts = append(content.Content.OfArrayOfContentParts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: part.Text,
						},
					})
				case PartTypeToolInvocation:
					if part.ToolInvocation == nil {
						return nil, fmt.Errorf("assistant message part has type tool-invocation but nil ToolInvocation field (ID: %s)", message.ID)
					}
					argsJSON, err := json.Marshal(part.ToolInvocation.Args)
					if err != nil {
						return nil, fmt.Errorf("marshalling tool input for call %s: %w", part.ToolInvocation.ToolCallID, err)
					}
					content.ToolCalls = append(content.ToolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: part.ToolInvocation.ToolCallID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      part.ToolInvocation.ToolName,
							Arguments: string(argsJSON),
						},
					})

					if part.ToolInvocation.State != ToolInvocationStateResult {
						continue
					}

					openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
						OfAssistant: content,
					})
					content = &openai.ChatCompletionAssistantMessageParam{}

					parts := []openai.ChatCompletionContentPartTextParam{}

					resultParts, err := toolResultToParts(part.ToolInvocation.Result)
					if err != nil {
						return nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
					}
					for _, resultPart := range resultParts {
						switch resultPart.Type {
						case PartTypeText:
							parts = append(parts, openai.ChatCompletionContentPartTextParam{
								Text: resultPart.Text,
							})
						case PartTypeFile:
							// Unfortunately, OpenAI doesn't support file content in tool messages.
							parts = append(parts, openai.ChatCompletionContentPartTextParam{
								Text: "File content was provided as a tool result, but is not supported by OpenAI.",
							})
							continue
						}
					}

					openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							ToolCallID: part.ToolInvocation.ToolCallID,
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfArrayOfContentParts: parts,
							},
						},
					})
				}
			}

			if len(content.Content.OfArrayOfContentParts) > 0 {
				openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfArrayOfContentParts: content.Content.OfArrayOfContentParts,
						},
					},
				})
			}
		}
	}

	return openaiMessages, nil
}

// OpenAIToDataStream pipes an OpenAI stream to a DataStream.
func OpenAIToDataStream(stream *ssestream.Stream[openai.ChatCompletionChunk]) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		var lastChunk *openai.ChatCompletionChunk
		var currentToolCallID string

		if stream.Err() != nil {
			if !yield(ErrorStreamPart{Content: stream.Err().Error()}, nil) {
				return
			}
		}

		for stream.Next() {
			chunk := stream.Current()
			lastChunk = &chunk

			if len(chunk.Choices) == 0 {
				break
			}
			choice := chunk.Choices[0]

			if choice.Delta.Content != "" {
				// Yield a Part object instead of TextStreamPart
				if !yield(TextStreamPart{Content: choice.Delta.Content}, nil) {
					return
				}
			}

			for _, toolCallDelta := range choice.Delta.ToolCalls {
				// The tool call ID is only present in the first delta.
				if toolCallDelta.ID != "" {
					currentToolCallID = toolCallDelta.ID // Update current ID when starting new tool call
					if !yield(ToolCallStartStreamPart{
						ToolCallID: currentToolCallID,
						ToolName:   toolCallDelta.Function.Name,
					}, nil) {
						return
					}
				}

				// Only emit delta parts if we have arguments
				if toolCallDelta.Function.Arguments != "" {
					if currentToolCallID == "" {
						if !yield(nil, fmt.Errorf("received tool call delta with empty ID and no current tool call")) {
							return
						}
						continue
					}
					if !yield(ToolCallDeltaStreamPart{
						ToolCallID:    currentToolCallID,
						ArgsTextDelta: toolCallDelta.Function.Arguments,
					}, nil) {
						return
					}
				}
			}

			if choice.FinishReason != "" {
				var finishReason FinishReason
				switch choice.FinishReason {
				case "tool_calls":
					finishReason = FinishReasonToolCalls
				default:
					finishReason = FinishReasonStop
				}
				if !yield(FinishStepStreamPart{
					IsContinued:  false,
					FinishReason: finishReason,
				}, nil) {
					return
				}
			}
		}

		var finishReason FinishReason
		var promptTokens, completionTokens *int64

		if lastChunk != nil && len(lastChunk.Choices) > 0 {
			choice := lastChunk.Choices[0]

			switch choice.FinishReason {
			case "tool_calls":
				finishReason = FinishReasonToolCalls
			default:
				finishReason = FinishReasonStop
			}

			if lastChunk.Usage.JSON.CompletionTokens.Valid() {
				tokens := int64(lastChunk.Usage.CompletionTokens)
				completionTokens = &tokens
			}
			if lastChunk.Usage.JSON.PromptTokens.Valid() {
				tokens := int64(lastChunk.Usage.PromptTokens)
				promptTokens = &tokens
			}
		}

		yield(FinishMessageStreamPart{
			FinishReason: finishReason,
			Usage: Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
			},
		}, nil)
	}
}
