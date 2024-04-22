export GEMINI_ENDPOINT=https://generativelanguage.googleapis.com
export GEMINI_API_KEY=YOUR_KEY
export GEMINI_MODEL=gemini-1.5-pro-latest
export VERSION=v1.1.0
rm -rf bin
mkdir bin
export GOOS=linux
export GOARCH=amd64
ls -al bin
/snap/bin/go build -trimpath -ldflags "-s -w" -o bin/google-gemini-openai-proxy .
ls -al bin
echo "using url $GEMINI_ENDPOINT"
echo "using key $GEMINI_API_KEY"
echo "using model $GEMINI_MODEL"
#/snap/bin/go build -o gemini main.go
bin/google-gemini-openai-proxy
