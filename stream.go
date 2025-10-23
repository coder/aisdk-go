package aisdk

import (
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
)

// Chat is the structure sent from `useChat` to the server.
// This can be extended if you'd like to send additional data with `body`.
type Chat struct {
	ID       string    `json:"id"`
	Messages []Message `json:"messages"`
}

// DataStream is a stream of DataStreamParts.
type DataStream iter.Seq2[DataStreamPart, error]

// WithToolCalling passes tool calls to the handleToolCall function.
func (s DataStream) WithToolCalling(handleToolCall func(toolCall ToolCall) any) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		// Track partial tool calls by ID
		partialToolCalls := make(map[string]struct {
			text     string
			step     int
			toolName string
		})

		// Track current step
		step := 0

		// Process a complete tool call
		processToolCall := func(id string, name string, args map[string]any) bool {
			if !yield(ToolCallStreamPart{
				ToolCallID: id,
				ToolName:   name,
				Args:       args,
			}, nil) {
				return false
			}

			// Call the handler and get the result
			result := handleToolCall(ToolCall{
				ID:   id,
				Name: name,
				Args: args,
			})

			// Yield the result
			return yield(ToolResultStreamPart{
				ToolCallID: id,
				Result:     result,
			}, nil)
		}

		// Process a tool call delta
		processDelta := func(id string, delta string) bool {
			partialCall := partialToolCalls[id]
			partialCall.text += delta
			partialToolCalls[id] = partialCall

			// Try to parse the partial JSON
			var args map[string]any
			if err := json.Unmarshal([]byte(partialCall.text), &args); err == nil {
				// Successfully parsed complete args, process the call
				if !processToolCall(id, partialCall.toolName, args) {
					return false
				}
				delete(partialToolCalls, id)
			}

			return true
		}

		for part, err := range s {
			if err != nil {
				yield(nil, err)
				return
			}

			if !yield(part, nil) {
				return
			}

			switch p := part.(type) {
			case StartStepStreamPart:
				// Keep track of step for tool calls
				step++

			case ToolCallStartStreamPart:
				// Initialize a new partial tool call
				partialToolCalls[p.ToolCallID] = struct {
					text     string
					step     int
					toolName string
				}{
					text:     "",
					step:     step,
					toolName: p.ToolName,
				}

			case ToolCallDeltaStreamPart:
				if !processDelta(p.ToolCallID, p.ArgsTextDelta) {
					return
				}

			case ToolCallStreamPart:
				if !processToolCall(p.ToolCallID, p.ToolName, p.Args) {
					return
				}
				delete(partialToolCalls, p.ToolCallID)

			case FinishStepStreamPart:
				// Clean up any remaining partial tool calls
				for id := range partialToolCalls {
					delete(partialToolCalls, id)
				}
			}
		}
	}
}

// WithAccumulator passes parts to the accumulator which aggregates them into a single message.
func (s DataStream) WithAccumulator(accumulator *DataStreamAccumulator) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		for part, err := range s {
			if err != nil {
				yield(nil, err)
				return
			}
			err = accumulator.Push(part)
			if err != nil {
				yield(nil, err)
				return
			}
			yield(part, nil)
		}
	}
}

// Pipe iterates over the DataStream and writes the parts to the writer.
func (s DataStream) Pipe(w io.Writer) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		flusher = nil
	}

	var pipeErr error
	s(func(part DataStreamPart, err error) bool {
		if err != nil {
			pipeErr = err
			return false
		}
		formatted, err := part.Format()
		if err != nil {
			pipeErr = err
			return false
		}
		_, err = fmt.Fprint(w, formatted)
		if err != nil {
			pipeErr = err
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	})
	return pipeErr
}

// DataStreamPart represents a part of the Vercel AI SDK data stream.
type DataStreamPart interface {
	Format() (string, error)
	TypeID() byte
}

// TextStreamPart corresponds to TYPE_ID '0'.
type TextStreamPart struct {
	Content string
}

func (p TextStreamPart) TypeID() byte { return '0' }
func (p TextStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal text content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// ReasoningStreamPart corresponds to TYPE_ID 'g'.
type ReasoningStreamPart struct {
	Content string
}

func (p ReasoningStreamPart) TypeID() byte { return 'g' }
func (p ReasoningStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal reasoning content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// RedactedReasoningStreamPart corresponds to TYPE_ID 'i'.
type RedactedReasoningStreamPart struct {
	Data string `json:"data"`
}

func (p RedactedReasoningStreamPart) TypeID() byte { return 'i' }
func (p RedactedReasoningStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ReasoningSignatureStreamPart corresponds to TYPE_ID 'j'.
type ReasoningSignatureStreamPart struct {
	Signature string `json:"signature"`
}

func (p ReasoningSignatureStreamPart) TypeID() byte { return 'j' }
func (p ReasoningSignatureStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// SourceStreamPart corresponds to TYPE_ID 'h'.
type SourceStreamPart struct {
	SourceType string `json:"sourceType"`
	ID         string `json:"id"`
	URL        string `json:"url"`
	Title      string `json:"title"`
}

func (p SourceStreamPart) TypeID() byte { return 'h' }
func (p SourceStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// FileStreamPart corresponds to TYPE_ID 'k'.
type FileStreamPart struct {
	Data     []byte `json:"data"`
	MimeType string `json:"mimeType"`
}

func (p FileStreamPart) TypeID() byte { return 'k' }
func (p FileStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// DataStreamDataPart corresponds to TYPE_ID '2'.
type DataStreamDataPart struct {
	Content []any
}

func (p DataStreamDataPart) TypeID() byte { return '2' }
func (p DataStreamDataPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal data content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// MessageAnnotationStreamPart corresponds to TYPE_ID '8'.
type MessageAnnotationStreamPart struct {
	Content []any
}

func (p MessageAnnotationStreamPart) TypeID() byte { return '8' }
func (p MessageAnnotationStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal annotation content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// ErrorStreamPart corresponds to TYPE_ID '3'.
type ErrorStreamPart struct {
	Content string
}

func (p ErrorStreamPart) TypeID() byte { return '3' }
func (p ErrorStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal error content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// ToolCall represents a tool call *request*.
type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type ToolCallResult interface {
	Part | []Part | any
}

// ToolCallStartStreamPart corresponds to TYPE_ID 'b'.
type ToolCallStartStreamPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
}

func (p ToolCallStartStreamPart) TypeID() byte { return 'b' }
func (p ToolCallStartStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ToolCallDeltaStreamPart corresponds to TYPE_ID 'c'.
type ToolCallDeltaStreamPart struct {
	ToolCallID    string `json:"toolCallId"`
	ArgsTextDelta string `json:"argsTextDelta"`
}

func (p ToolCallDeltaStreamPart) TypeID() byte { return 'c' }
func (p ToolCallDeltaStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ToolCallStreamPart corresponds to TYPE_ID '9'.
type ToolCallStreamPart struct {
	ToolCallID string         `json:"toolCallId"`
	ToolName   string         `json:"toolName"`
	Args       map[string]any `json:"args"`
}

func (p ToolCallStreamPart) TypeID() byte { return '9' }
func (p ToolCallStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ToolResultStreamPart corresponds to TYPE_ID 'a'.
type ToolResultStreamPart struct {
	ToolCallID string `json:"toolCallId"`
	Result     any    `json:"result"`
}

func (p ToolResultStreamPart) TypeID() byte { return 'a' }
func (p ToolResultStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// StartStepStreamPart corresponds to TYPE_ID 'f'.
type StartStepStreamPart struct {
	MessageID string `json:"messageId"`
}

func (p StartStepStreamPart) TypeID() byte { return 'f' }
func (p StartStepStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// FinishReason defines the possible reasons for finishing a step or message.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonError         FinishReason = "error"
	FinishReasonOther         FinishReason = "other"
	FinishReasonUnknown       FinishReason = "unknown"
)

// Usage details the token usage.
type Usage struct {
	PromptTokens     *int64 `json:"promptTokens"`
	CompletionTokens *int64 `json:"completionTokens"`
}

// FinishStepStreamPart corresponds to TYPE_ID 'e'.
type FinishStepStreamPart struct {
	FinishReason FinishReason `json:"finishReason"`
	Usage        Usage        `json:"usage"`
	IsContinued  bool         `json:"isContinued"`
}

func (p FinishStepStreamPart) TypeID() byte { return 'e' }
func (p FinishStepStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// FinishMessageStreamPart corresponds to TYPE_ID 'd'.
type FinishMessageStreamPart struct {
	FinishReason FinishReason `json:"finishReason"`
	Usage        Usage        `json:"usage"`
}

func (p FinishMessageStreamPart) TypeID() byte { return 'd' }
func (p FinishMessageStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

func formatJSONPart(part DataStreamPart) (string, error) {
	jsonData, err := json.Marshal(part)
	if err != nil {
		return "", fmt.Errorf("failed to marshal part type %T: %w", part, err)
	}
	return fmt.Sprintf("%c:%s\n", part.TypeID(), string(jsonData)), nil
}

type Attachment struct {
	Name        string `json:"name,omitempty"`
	ContentType string `json:"contentType,omitempty"`
	URL         string `json:"url"`
}

type Message struct {
	ID          string           `json:"id"`
	CreatedAt   *json.RawMessage `json:"createdAt,omitempty"`
	Content     string           `json:"content"`
	Role        string           `json:"role"`
	Parts       []Part           `json:"parts,omitempty"`
	Annotations []any            `json:"annotations,omitempty"`
	Attachments []Attachment     `json:"experimental_attachments,omitempty"`
}

type PartType string

const (
	PartTypeText           PartType = "text"
	PartTypeReasoning      PartType = "reasoning"
	PartTypeToolInvocation PartType = "tool-invocation"
	PartTypeSource         PartType = "source"
	PartTypeFile           PartType = "file"
	PartTypeStepStart      PartType = "step-start"
)

type ReasoningDetail struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

type SourceInfo struct {
	URI         string         `json:"uri,omitempty"`
	ContentType string         `json:"contentType,omitempty"`
	Data        string         `json:"data,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type Part struct {
	Type PartType `json:"type"`

	// Type: "text"
	Text string `json:"text,omitempty"`

	// Type: "reasoning"
	Reasoning string            `json:"reasoning,omitempty"`
	Details   []ReasoningDetail `json:"details,omitempty"`

	// Type: "tool-invocation"
	ToolInvocation *ToolInvocation `json:"toolInvocation,omitempty"`

	// Type: "source"
	Source *SourceInfo `json:"source,omitempty"`

	// Type: "file"
	MimeType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`

	// Type: "step-start" - No additional fields

	isComplete bool // Internal accumulator tracking
}

type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Schema      Schema `json:"parameters"`
}

type Schema struct {
	Required   []string       `json:"required"`
	Properties map[string]any `json:"properties"`
}

type ToolInvocationState string

const (
	ToolInvocationStateCall        ToolInvocationState = "call"
	ToolInvocationStatePartialCall ToolInvocationState = "partial-call"
	ToolInvocationStateResult      ToolInvocationState = "result"
)

type ToolInvocation struct {
	State      ToolInvocationState `json:"state"`
	Step       *int                `json:"step,omitempty"`
	ToolCallID string              `json:"toolCallId"`
	ToolName   string              `json:"toolName"`
	Args       any                 `json:"args"`
	Result     any                 `json:"result,omitempty"`
}

func WriteDataStreamHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("X-Vercel-AI-Data-Stream", "v1")
	w.WriteHeader(http.StatusOK)
}

// DataStreamAccumulator accumulates DataStreamParts into Messages.
type DataStreamAccumulator struct {
	messages       []Message
	currentMessage *Message
	wipToolCalls   map[string]*Part // Keyed by ToolCallID, points to Part in currentMessage.Parts
	finishReason   FinishReason
	usage          Usage
}

func (a *DataStreamAccumulator) ensureCurrentMessage() {
	if a.currentMessage == nil {
		a.currentMessage = &Message{
			Role:  "assistant",
			Parts: make([]Part, 0, 5),
		}
		a.wipToolCalls = make(map[string]*Part)
	}
}

func (a *DataStreamAccumulator) findPart(toolCallID string) *Part {
	if a.currentMessage == nil {
		return nil
	}
	for i := range a.currentMessage.Parts {
		if a.currentMessage.Parts[i].Type == PartTypeToolInvocation &&
			a.currentMessage.Parts[i].ToolInvocation != nil &&
			a.currentMessage.Parts[i].ToolInvocation.ToolCallID == toolCallID {
			return &a.currentMessage.Parts[i]
		}
	}
	return nil
}

func (a *DataStreamAccumulator) Push(part DataStreamPart) error {
	if _, isFinal := part.(FinishMessageStreamPart); !isFinal {
		a.ensureCurrentMessage()
	}

	var currentMsgPtr *Message
	if a.currentMessage != nil {
		currentMsgPtr = a.currentMessage
	}

	switch p := part.(type) {
	case TextStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add TextStreamPart without an active message")
		}
		currentMsgPtr.Content += p.Content
		numParts := len(currentMsgPtr.Parts)
		if numParts > 0 && currentMsgPtr.Parts[numParts-1].Type == PartTypeText {
			currentMsgPtr.Parts[numParts-1].Text += p.Content
		} else {
			currentMsgPtr.Parts = append(currentMsgPtr.Parts, Part{
				Type: PartTypeText,
				Text: p.Content,
			})
		}

	case ReasoningStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add ReasoningStreamPart without an active message")
		}
		var reasoningPart *Part
		for i := range currentMsgPtr.Parts {
			if currentMsgPtr.Parts[i].Type == PartTypeReasoning {
				reasoningPart = &currentMsgPtr.Parts[i]
				break
			}
		}
		if reasoningPart == nil {
			currentMsgPtr.Parts = append(currentMsgPtr.Parts, Part{Type: PartTypeReasoning})
			reasoningPart = &currentMsgPtr.Parts[len(currentMsgPtr.Parts)-1]
		}
		reasoningPart.Reasoning += p.Content

	case FileStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add FileStreamPart without an active message")
		}
		currentMsgPtr.Parts = append(currentMsgPtr.Parts, Part{
			Type:     PartTypeFile,
			MimeType: p.MimeType,
			Data:     p.Data,
		})

	case SourceStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add SourceStreamPart without an active message")
		}
		currentMsgPtr.Parts = append(currentMsgPtr.Parts, Part{
			Type: PartTypeSource,
			Source: &SourceInfo{
				URI:         p.URL,
				ContentType: "",
				Data:        "",
				Metadata:    map[string]any{"id": p.ID, "title": p.Title, "sourceType": p.SourceType},
			},
		})

	case StartStepStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("StartStepStreamPart received before message initialization")
		}
		if currentMsgPtr.ID == "" {
			currentMsgPtr.ID = p.MessageID
		}
		currentMsgPtr.Parts = append(currentMsgPtr.Parts, Part{Type: PartTypeStepStart})

	case ToolCallStartStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add ToolCallStartStreamPart without an active message")
		}
		// Initialize a new tool call
		newPart := Part{
			Type: PartTypeToolInvocation,
			ToolInvocation: &ToolInvocation{
				State:      ToolInvocationStatePartialCall,
				ToolCallID: p.ToolCallID,
				ToolName:   p.ToolName,
				Args:       "",
			},
			isComplete: false,
		}
		currentMsgPtr.Parts = append(currentMsgPtr.Parts, newPart)
		a.wipToolCalls[p.ToolCallID] = &currentMsgPtr.Parts[len(currentMsgPtr.Parts)-1]

	case ToolCallDeltaStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add ToolCallDeltaStreamPart without an active message")
		}
		wipCallPart, exists := a.wipToolCalls[p.ToolCallID]
		if exists && wipCallPart.ToolInvocation != nil {
			if argsStr, ok := wipCallPart.ToolInvocation.Args.(string); ok {
				wipCallPart.ToolInvocation.Args = argsStr + p.ArgsTextDelta
			} else {
				return fmt.Errorf("tool call delta received for non-string args (ID: %s)", p.ToolCallID)
			}
		}

	case ToolCallStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add ToolCallStreamPart without an active message")
		}
		// Update or create tool call
		existingPart := a.findPart(p.ToolCallID)
		if existingPart != nil && existingPart.ToolInvocation != nil {
			existingPart.ToolInvocation.ToolName = p.ToolName
			existingPart.ToolInvocation.Args = p.Args
			existingPart.ToolInvocation.State = ToolInvocationStateCall
			existingPart.isComplete = true
		} else {
			currentMsgPtr.Parts = append(currentMsgPtr.Parts, Part{
				Type: PartTypeToolInvocation,
				ToolInvocation: &ToolInvocation{
					State:      ToolInvocationStateCall,
					ToolCallID: p.ToolCallID,
					ToolName:   p.ToolName,
					Args:       p.Args,
				},
				isComplete: true,
			})
		}
		delete(a.wipToolCalls, p.ToolCallID)

	case ToolResultStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add ToolResultStreamPart without an active message")
		}
		// Find and update existing tool call with result
		existingPart := a.findPart(p.ToolCallID)
		if existingPart != nil && existingPart.ToolInvocation != nil {
			existingPart.ToolInvocation.State = ToolInvocationStateResult
			existingPart.ToolInvocation.Result = p.Result
		} else {
			return fmt.Errorf("tool result received for unknown tool call ID: %s", p.ToolCallID)
		}

	case DataStreamDataPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add DataStreamDataPart without an active message")
		}
		currentMsgPtr.Annotations = append(currentMsgPtr.Annotations, p.Content...)

	case MessageAnnotationStreamPart:
		if currentMsgPtr == nil {
			return fmt.Errorf("cannot add MessageAnnotationStreamPart without an active message")
		}
		currentMsgPtr.Annotations = append(currentMsgPtr.Annotations, p.Content...)

	case FinishStepStreamPart:
		if currentMsgPtr != nil {
			// Clean up any remaining WIP tool calls
			for id, wipCallPart := range a.wipToolCalls {
				if !wipCallPart.isComplete && wipCallPart.ToolInvocation != nil {
					if argsStr, ok := wipCallPart.ToolInvocation.Args.(string); ok && argsStr != "" {
						var parsedArgs map[string]any
						if json.Unmarshal([]byte(argsStr), &parsedArgs) == nil {
							wipCallPart.ToolInvocation.Args = parsedArgs
							wipCallPart.ToolInvocation.State = ToolInvocationStateCall
						}
					}
					wipCallPart.isComplete = true
				}
				delete(a.wipToolCalls, id)
			}

			if !p.IsContinued {
				a.messages = append(a.messages, *currentMsgPtr)
				a.currentMessage = nil
				a.wipToolCalls = nil
			}
		}
		a.finishReason = p.FinishReason

	case FinishMessageStreamPart:
		if currentMsgPtr != nil {
			// Clean up any remaining WIP tool calls
			for _, wipCallPart := range a.wipToolCalls {
				if !wipCallPart.isComplete && wipCallPart.ToolInvocation != nil {
					if argsStr, ok := wipCallPart.ToolInvocation.Args.(string); ok && argsStr != "" {
						var parsedArgs map[string]any
						if json.Unmarshal([]byte(argsStr), &parsedArgs) == nil {
							wipCallPart.ToolInvocation.Args = parsedArgs
							wipCallPart.ToolInvocation.State = ToolInvocationStateCall
						}
					}
					wipCallPart.isComplete = true
				}
			}
			a.messages = append(a.messages, *currentMsgPtr)
		}
		a.finishReason = p.FinishReason
		a.currentMessage = nil
		a.wipToolCalls = nil
		a.usage = p.Usage

	case ErrorStreamPart:
		a.finishReason = FinishReasonError
		return fmt.Errorf("error in stream: %s", p.Content)

	case RedactedReasoningStreamPart, ReasoningSignatureStreamPart:
		// No action needed for accumulation

	default:
		return fmt.Errorf("unhandled part type: %T", part)
	}

	return nil
}

func (a *DataStreamAccumulator) Messages() []Message {
	return a.messages
}

func (a *DataStreamAccumulator) FinishReason() FinishReason {
	return a.finishReason
}

func (a *DataStreamAccumulator) Usage() Usage {
	return a.usage
}

func toolResultToParts(result any) ([]Part, error) {
	switch r := result.(type) {
	case []Part:
		return r, nil
	case Part:
		return []Part{r}, nil
	default:
		jsonData, err := json.Marshal(r)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool call result: %w", err)
		}
		return []Part{{Type: PartTypeText, Text: string(jsonData)}}, nil
	}
}
