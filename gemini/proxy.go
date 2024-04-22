package gemini

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"regexp"
	"strings"
    "strconv"
	"time"

	
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
	"github.com/soulteary/google-gemini-openai-proxy/define"
	"github.com/soulteary/google-gemini-openai-proxy/util"
)

func ProxyWithConverter(requestConverter RequestConverter) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method == http.MethodOptions {
			c.Header("Access-Control-Allow-Origin", "*")
			c.Header("Access-Control-Allow-Methods", "GET, OPTIONS, POST")
			c.Header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-requested-with")
			c.Status(200)
			return
		}
		Proxy(c, requestConverter)
	}
}
// InputPayload represents the top-level structure of the input JSON.
type InputPayload struct {
    Candidates []Candidate `json:"candidates"`
}

// Candidate holds the individual candidate data in the payload.
type Candidate struct {
    Content      Content       `json:"content"`
    FinishReason string        `json:"finishReason"`
    Index        int           `json:"index"`
    SafetyRatings []SafetyRating `json:"safetyRatings"`
}

// Content holds the text content and role.
type Content struct {
    Parts []Part `json:"parts"`
    Role  string `json:"role"`
}

// Part holds individual sections of content.
type Part struct {
    Text string `json:"text"`
}

// SafetyRating holds safety ratings concerning various categories.
type SafetyRating struct {
    Category    string `json:"category"`
    Probability string `json:"probability"`
}
// OutPayload defines the structure for the output JSON.
type OutPayload struct {
    Choices []struct {
        FinishReason string `json:"finish_reason"`
        Index        int    `json:"index"`
        Message struct {
            Content string `json:"content"`
            Role    string `json:"role"`
        } `json:"message"`
        LogProbs interface{} `json:"logprobs"` // Assuming null can be represented as interface{} type
    } `json:"choices"`
    Created uint64 `json:"created"`
    ID      string `json:"id"`
    Model   string `json:"model"`
    Object  string `json:"object"`
    Usage   struct {
        CompletionTokens int `json:"completion_tokens"`
        PromptTokens     int `json:"prompt_tokens"`
        TotalTokens      int `json:"total_tokens"`
    } `json:"usage"`
}

// transformInputToOutput converts the input payload into the output payload.
func transformInputToOutput(input InputPayload) OutPayload {
    var out OutPayload
    out.Created = uint64(time.Now().Unix())
    out.ID = "unique-id-based-on-context" // This should be dynamically generated or passed in
    out.Model = "model-used-for-processing" // This should reflect the actual model used
    out.Object = "type-of-operation" // E.g., "text_completion"

    for _, candidate := range input.Candidates {
        choice := struct {
            FinishReason string `json:"finish_reason"`
            Index        int    `json:"index"`
            Message struct {
                Content string `json:"content"`
                Role    string `json:"role"`
            } `json:"message"`
            LogProbs interface{} `json:"logprobs"`
        }{
            FinishReason: strings.ToLower(candidate.FinishReason),
            Index:        candidate.Index,
            Message: struct {
                Content string `json:"content"`
                Role    string `json:"role"`
            }{
                Content: candidate.Content.Parts[0].Text,
                Role:    "assistant", // Changed from "model" based on your specification
            },
        }
        out.Choices = append(out.Choices, choice)
    }

    return out
}

var maskURL = regexp.MustCompile(`key=.+`)

// Proxy Azure OpenAI
// Define the Proxy function that sets up a reverse proxy for handling HTTP requests and responses.
func Proxy(c *gin.Context, requestConverter RequestConverter) {
    // The director function configures the request before it is forwarded.
    director := func(req *http.Request) {
        // Check if the request body is empty and send an error message if it is.
        if req.Body == nil {
            util.SendError(c, errors.New("request body is empty"))
            return
        }
        // Read all data from the request body.
        body, _ := io.ReadAll(req.Body)
        // Reset the request body so it can be read again during forwarding.
        req.Body = io.NopCloser(bytes.NewBuffer(body))

        // Declare a variable to hold the payload according to the OpenAIPayload structure.
        var openaiPayload OpenAIPayload

        // Unmarshal the JSON body into the openaiPayload variable; send an error if it fails.
        if err := json.Unmarshal(body, &openaiPayload); err != nil {
            util.SendError(c, errors.Wrap(err, "parse payload error"))
            return
        }

        // Extract the model name from the payload and use a default if it's empty.
        model := strings.TrimSpace(openaiPayload.Model)
        if model == "" {
            model = define.DEFAULT_REST_API_MODEL_NAME
        }

        // Initialize a payload variable to configure it for the API call.
        var payload GoogleGeminiPayload
        // Loop through messages in the payload to adjust roles and format parts.
        for _, data := range openaiPayload.Messages {
            var message GeminiPayloadContents
            if strings.ToLower(data.Role) == "user" {
                message.Role = "user"
            } else {
                message.Role = "model"
            }
            message.Parts = append(message.Parts, GeminiPayloadParts{
                Text: strings.TrimSpace(data.Content),
            })
            payload.Contents = append(payload.Contents, message)
        }

        // Set default safety settings for the content.
        var safetySettings []GeminiSafetySettings
        safetySettings = append(safetySettings, GeminiSafetySettings{
            Category:  "HARM_CATEGORY_DANGEROUS_CONTENT",
            Threshold: "BLOCK_NONE",
        })
        payload.SafetySettings = safetySettings

        // Configure generation settings based on the original payload.
        payload.GenerationConfig.StopSequences = []string{"Title"}
        payload.GenerationConfig.Temperature = openaiPayload.Temperature
        payload.GenerationConfig.MaxOutputTokens = openaiPayload.MaxTokens
        payload.GenerationConfig.TopP = openaiPayload.TopP
        // payload.GenerationConfig.TopK = openaiPayload.TopK

        // Retrieve deployment settings based on the model.
        deployment, err := GetDeploymentByModel(model)
        if err != nil {
            util.SendError(c, err)
            return
        }

        // Extract the API token from the header or deployment configuration.
        token := deployment.ApiKey
        if token == "" {
            rawToken := req.Header.Get("Authorization")
            token = strings.TrimPrefix(rawToken, "Bearer ")
        }
        if token == "" {
            util.SendError(c, errors.New("token is empty"))
            return
        }
        req.Header.Set("Authorization", token)
        req.Header.Set("ngrok-skip-browser-warning", "xxxxxxxx")

        // Convert the prepared payload into JSON format for forwarding.
        payloadJSON, err := json.Marshal(payload)
        if err != nil {
            util.SendError(c, errors.Wrap(err, "Error converting to JSON"))
            return
        }

        // Log the URL being requested.
        originURL := req.URL.String()
		if requestConverter != nil {
			req, err = requestConverter.Convert(req, deployment, payloadJSON)
			if err != nil {
				util.SendError(c, errors.Wrap(err, "convert request error"))
				return
			}
		} else {
			util.SendError(c, errors.New("requestConverter is nil"))
			return
		}
		

        // Log the proxy operation detailing the request being forwarded.
        log.Printf("proxying request [%s] %s -> %s", model, originURL, maskURL.ReplaceAllString(req.URL.String(), "key=******"))
        log.Printf("proxying request [%s] %s -> %s", model, originURL, req.URL.String())
    }

    // Initialize the Reverse Proxy
    proxy := &httputil.ReverseProxy{
        Director: director,
        ModifyResponse: func(resp *http.Response) error {
            // Read the original response body
            originalBody, err := io.ReadAll(resp.Body)
            if err != nil {
				
                return errors.Wrap(err, "failed to read response body")
            }
            defer resp.Body.Close()

            // Unmarshal the response body into your structured type
            var inputPayload InputPayload
            if err := json.Unmarshal(originalBody, &inputPayload); err != nil {
                return errors.Wrap(err, "failed to unmarshal response body")
            }

            // Transform the response here using the function you've defined
            outPayload := transformInputToOutput(inputPayload)

            // Marshal the modified response body back to JSON
            modifiedBody, err := json.Marshal(outPayload)
            if err != nil {
                return errors.Wrap(err, "failed to marshal modified response body")
            }

            // Set the modified body as the new response body
            resp.Body = io.NopCloser(bytes.NewBuffer(modifiedBody))
            resp.ContentLength = int64(len(modifiedBody))
            resp.Header.Set("Content-Length", strconv.Itoa(len(modifiedBody)))

            return nil
        },
    }

    // Set up the transport and serve the proxy
    transport, err := util.NewProxyFromEnv()
    if err != nil {
        util.SendError(c, errors.Wrap(err, "get proxy error"))
        return
    }
	if transport != nil {
		proxy.Transport = transport
	} else {
		// Set to default transport if custom transport failed to initialize
		proxy.Transport = http.DefaultTransport
	}
	
    proxy.ServeHTTP(c.Writer, c.Request)
}

func GetDeploymentByModel(model string) (*DeploymentConfig, error) {
	deploymentConfig, exist := ModelDeploymentConfig[model]
	if !exist {
		return nil, errors.New(fmt.Sprintf("deployment config for %s not found", model))
	}
	return &deploymentConfig, nil
}
