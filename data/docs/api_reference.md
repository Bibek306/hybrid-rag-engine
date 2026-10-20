# Developer API Reference Guide

## Authentication and Security
All API calls must include an Authorization header using Bearer token format:
`Authorization: Bearer <JWT_TOKEN>`. 
Access tokens expire after 3600 seconds (1 hour). When a token expires, the server returns an `HTTP 401 Unauthorized` response with the body `{"error": "token_expired"}`. Use the `/v1/auth/refresh` endpoint with your long-lived refresh token to acquire a fresh access token without requiring user re-authentication.

## Rate Limiting Policies
The API imposes a strict rate limit of 100 requests per minute per IP address. When an application exceeds this threshold, the API responds with `HTTP 429 Too Many Requests`. The response includes a `Retry-After` header indicating the number of seconds the client must wait before retrying the call. Clients should implement exponential backoff with jitter to prevent thundering herd problems.

## Idempotency Keys
For mutating actions like `/v1/charges` or `/v1/transfers`, clients must provide an `Idempotency-Key` header with a unique UUIDv4. If a network interruption occurs, re-sending the request with the identical key returns the cached response rather than creating duplicate transactions.