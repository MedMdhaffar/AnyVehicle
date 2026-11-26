import assert from 'node:assert/strict'
import { Readable, Writable } from 'node:stream'
import { finished } from 'node:stream/promises'
import test from 'node:test'
import { forwardApiResponse } from '../apiProxy.js'

for (const status of [202, 400, 409, 500]) {
  test(`proxy preserves HTTP ${status} and the complete JSON body`, async () => {
    const body = Buffer.from(JSON.stringify({ error: 'Video not found · vidéo' }))
    const upstream = Readable.from([body.subarray(0, 5), body.subarray(5)])
    upstream.statusCode = status
    upstream.headers = { 'content-type': 'application/json', 'content-length': String(body.length), connection: 'close' }
    const chunks = []
    const response = new Writable({ write(chunk, encoding, done) { chunks.push(chunk); done() } })
    response.writeHead = (code, headers) => {
      assert.equal(code, status)
      assert.equal(headers['content-type'], 'application/json')
      assert.equal(headers['content-length'], String(body.length))
      assert.equal(headers.connection, undefined)
    }
    forwardApiResponse(upstream, response)
    await finished(response)
    assert.deepEqual(Buffer.concat(chunks), body)
  })
}

test('proxy forwards binary crop bytes without buffering or JSON conversion', async () => {
  const body = Buffer.from([137, 80, 78, 71, 0, 255])
  const upstream = Readable.from([body])
  upstream.statusCode = 200
  upstream.headers = { 'content-type': 'image/png', 'transfer-encoding': 'chunked' }
  const chunks = []
  const response = new Writable({ write(chunk, encoding, done) { chunks.push(chunk); done() } })
  response.writeHead = (status, headers) => {
    assert.equal(status, 200)
    assert.equal(headers['content-type'], 'image/png')
    assert.equal(headers['transfer-encoding'], undefined)
  }
  forwardApiResponse(upstream, response)
  await finished(response)
  assert.deepEqual(Buffer.concat(chunks), body)
})
