import { pipeline } from 'node:stream'

// Forward error bodies as well as successful JSON, images and streaming uploads.
export function forwardApiResponse(upstream, response) {
  const headers = { ...upstream.headers }
  delete headers.connection
  delete headers['keep-alive']
  delete headers['transfer-encoding']
  response.writeHead(upstream.statusCode || 502, headers)
  pipeline(upstream, response, () => {})
}
