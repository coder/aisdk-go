package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// GoogleStreamIterator defines the interface for iterating over Google AI stream responses.
// This allows for mocking in tests.
type GoogleStreamIterator interface {
	Next() (*genai.GenerateContentResponse, error)
}

func ToolsToGoogle(tools []Tool) ([]*genai.Tool, error) {
	functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(tools))

	var propertyToSchema func(property map[string]any) (*genai.Schema, error)
	propertyToSchema = func(property map[string]any) (*genai.Schema, error) {
		schema := &genai.Schema{}

		typeRaw, ok := property["type"]
		if ok {
			typ, ok := typeRaw.(string)
			if !ok {
				return nil, fmt.Errorf("type is not a string: %T", typeRaw)
			}
			schema.Type = genai.Type(strings.ToUpper(typ))
		}

		descriptionRaw, ok := property["description"]
		if ok {
			description, ok := descriptionRaw.(string)
			if !ok {
				return nil, fmt.Errorf("description is not a string: %T", descriptionRaw)
			}
			schema.Description = description
		}

		propertiesRaw, ok := property["properties"]
		if ok {
			properties, ok := propertiesRaw.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("properties is not a map[string]any: %T", propertiesRaw)
			}
			for key, value := range properties {
				propMap, ok := value.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("property %q is not a map[string]any: %T", key, value)
				}
				subschema, err := propertyToSchema(propMap)
				if err != nil {
					return nil, fmt.Errorf("property %q has non-object properties: %w", key, err)
				}
				schema.Properties[key] = subschema
			}
		}

		itemsRaw, ok := property["items"]
		if ok {
			items, ok := itemsRaw.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("items is not a map[string]any: %T", itemsRaw)
			}
			subschema, err := propertyToSchema(items)
			if err != nil {
				return nil, fmt.Errorf("items has non-object properties: %w", err)
			}
			schema.Items = subschema
		}

		return schema, nil
	}

	for _, tool := range tools {
		var schema *genai.Schema
		if tool.Schema.Properties != nil {
			schema = &genai.Schema{
				Type:       genai.TypeObject,
				Properties: make(map[string]*genai.Schema),
				Required:   tool.Schema.Required,
			}

			for key, value := range tool.Schema.Properties {
				propMap, ok := value.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("property %q is not a map[string]any: %T", key, value)
				}
				subschema, err := propertyToSchema(propMap)
				if err != nil {
					return nil, fmt.Errorf("property %q has non-object properties: %w", key, err)
				}
				schema.Properties[key] = subschema
			}
		}

		functionDeclarations = append(functionDeclarations, &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  schema,
		})
	}
	return []*genai.Tool{{
		FunctionDeclarations: functionDeclarations,
	}}, nil
}

// MessagesToGoogle converts internal message format to Google's genai.Content slice.
// System messages are ignored.
func MessagesToGoogle(messages []Message) ([]*genai.Content, error) {
	googleContents := []*genai.Content{}

	for _, message := range messages {
		switch message.Role {
		case "system":
			// System messages are ignored for Google's main message history.
			// They are handled separately via SystemInstruction.

		case "user":
			content := &genai.Content{
				Role: "user",
			}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content.Parts = append(content.Parts, &genai.Part{Text: part.Text})
				case PartTypeFile:
					content.Parts = append(content.Parts, &genai.Part{InlineData: &genai.Blob{
						Data:     part.Data,
						MIMEType: part.MimeType,
					}})
				}
			}

			for _, attachment := range message.Attachments {
				parts := strings.SplitN(attachment.URL, ",", 2)
				if len(parts) != 2 {
					return nil, fmt.Errorf("invalid attachment URL: %s", attachment.URL)
				}
				decoded, err := base64.StdEncoding.DecodeString(parts[1])
				if err != nil {
					return nil, fmt.Errorf("failed to decode attachment: %w", err)
				}
				content.Parts = append(content.Parts, &genai.Part{InlineData: &genai.Blob{
					Data:     decoded,
					MIMEType: attachment.ContentType,
				}})
			}

			googleContents = append(googleContents, content)
		case "assistant":
			content := &genai.Content{
				Role: "model",
			}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content.Parts = append(content.Parts, &genai.Part{
						Text: part.Text,
					})
				case PartTypeToolInvocation:
					if part.ToolInvocation == nil {
						return nil, fmt.Errorf("assistant message part has type tool-invocation but nil ToolInvocation field (ID: %s)", message.ID)
					}
					argsMap, ok := part.ToolInvocation.Args.(map[string]any)
					if !ok && part.ToolInvocation.Args != nil { // Allow nil args
						return nil, fmt.Errorf("tool call args for %s are not map[string]any: %T", part.ToolInvocation.ToolName, part.ToolInvocation.Args)
					}
					fc := genai.FunctionCall{
						ID:   part.ToolInvocation.ToolCallID,
						Name: part.ToolInvocation.ToolName,
						Args: argsMap,
					}
					content.Parts = append(content.Parts, &genai.Part{
						FunctionCall: &fc,
					})

					if part.ToolInvocation.State != ToolInvocationStateResult {
						continue
					}

					googleContents = append(googleContents, content)
					content = &genai.Content{
						Role: "model",
					}

					googleParts := []*genai.Part{}

					parts, err := toolResultToParts(part.ToolInvocation.Result)
					if err != nil {
						return nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
					}
					for _, part := range parts {
						switch part.Type {
						case PartTypeText:
							googleParts = append(googleParts, &genai.Part{
								Text: part.Text,
							})
						case PartTypeFile:
							googleParts = append(googleParts, &genai.Part{
								InlineData: &genai.Blob{
									Data:     part.Data,
									MIMEType: part.MimeType,
								}},
							)
						}
					}

					fr := genai.FunctionResponse{
						Name:     part.ToolInvocation.ToolName,
						ID:       part.ToolInvocation.ToolCallID,
						Response: map[string]any{"output": googleParts},
					}
					content.Parts = append(content.Parts, &genai.Part{FunctionResponse: &fr})
				}
			}

			googleContents = append(googleContents, content)
		default:
			return nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return googleContents, nil
}

// GoogleToDataStream pipes a Google AI stream to a DataStream.
func GoogleToDataStream(stream iter.Seq2[*genai.GenerateContentResponse, error]) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		accumulatedContents := []*genai.Content{}
		finalReason := FinishReasonUnknown
		isNewMessageSegment := true
		var lastResp *genai.GenerateContentResponse

		for resp, err := range stream {
			if err != nil {
				yield(nil, err)
				return
			}

			if resp == nil {
				continue
			}

			lastResp = resp

			if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
				continue
			}
			cand := resp.Candidates[0]
			content := cand.Content

			if isNewMessageSegment {
				messageID := uuid.New().String()
				if !yield(StartStepStreamPart{
					MessageID: messageID,
				}, nil) {
					return
				}
				isNewMessageSegment = false
			}

			accumulatedContents = append(accumulatedContents, content)

			for _, part := range content.Parts {
				if part.FunctionCall != nil {
					fc := part.FunctionCall
					// Google's stream often sends the full FunctionCall in one part.
					// We translate this into start and delta parts for the generic stream.
					toolCallID := uuid.New().String()

					// Emit start part ('b')
					if !yield(ToolCallStartStreamPart{
						ToolCallID: toolCallID,
						ToolName:   fc.Name,
					}, nil) {
						return
					}

					// Convert args to JSON string and emit as delta part ('c')
					// WithToolCalling will accumulate this (potentially single) delta.
					argsJSON, err := json.Marshal(fc.Args)
					if err == nil {
						if !yield(ToolCallDeltaStreamPart{
							ToolCallID:    toolCallID,
							ArgsTextDelta: string(argsJSON),
						}, nil) {
							return
						}
					}

					finalReason = FinishReasonToolCalls
				} else if part.Text != "" {
					text := part.Text
					if part.Thought {
						if !yield(ReasoningStreamPart{Content: text}, nil) {
							return
						}
					} else {
						if !yield(TextStreamPart{Content: text}, nil) {
							return
						}
					}
				} // Add handling for other part types (e.g., FunctionResponse) if necessary
			}
		}

		// Determine the final reason only *after* the loop completes.
		var actualFinalReason FinishReason
		var finalUsage Usage
		var finalCand *genai.Candidate

		if lastResp != nil && len(lastResp.Candidates) > 0 {
			finalCand = lastResp.Candidates[0]
		}

		// Use the detected tool call reason if present, otherwise determine from candidate.
		if finalReason == FinishReasonToolCalls {
			actualFinalReason = FinishReasonToolCalls
		} else if finalCand != nil {
			switch finalCand.FinishReason {
			case genai.FinishReasonStop:
				actualFinalReason = FinishReasonStop
			case genai.FinishReasonMaxTokens:
				actualFinalReason = FinishReasonLength
			case genai.FinishReasonSafety:
				actualFinalReason = FinishReasonContentFilter
			case genai.FinishReasonRecitation:
				actualFinalReason = FinishReasonContentFilter // Treat recitation as content filter
			case genai.FinishReasonUnspecified:
				actualFinalReason = FinishReasonUnknown
			default:
				actualFinalReason = FinishReasonOther
			}
		} else {
			// If no candidate and no tool call detected, assume stop or error?
			// Let's default to Stop, assuming the stream ended normally without specific reason.
			actualFinalReason = FinishReasonStop
		}

		// Extract final usage data if available
		if lastResp != nil && lastResp.UsageMetadata != nil {
			if lastResp.UsageMetadata.PromptTokenCount > 0 {
				promptTokens := int64(lastResp.UsageMetadata.PromptTokenCount)
				finalUsage.PromptTokens = &promptTokens
			}
			if lastResp.UsageMetadata.CandidatesTokenCount > 0 {
				completionTokens := int64(lastResp.UsageMetadata.CandidatesTokenCount)
				finalUsage.CompletionTokens = &completionTokens
			}
		}

		// Send final finish step part
		if !yield(FinishStepStreamPart{
			FinishReason: actualFinalReason,
			Usage:        finalUsage,
			IsContinued:  false, // This is the final step
		}, nil) {
			return // Stop if yield fails
		}

		// Send final finish message part
		yield(FinishMessageStreamPart{
			FinishReason: actualFinalReason,
			Usage:        finalUsage,
		}, nil) // Ignore yield result here as we're at the very end
	}
}
