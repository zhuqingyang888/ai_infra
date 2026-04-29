# HTTP 和 JSON 基础

## 1. HTTP是什么？

HTTP（HyperText Transfer Protocol）是一种用于在互联网上传输数据的协议。它定义了客户端和服务器之间的通信规则。

基本流程是：

客户端向服务器发送request，服务器返回response。

## 2. request 是什么？

request 是客户端发给服务端的请求。

一个请求通常包含：

- 请求方法，比如 GET 或 POST
- 请求路径，比如 /ping 或 /predict
- 请求参数
- 请求体 body

## 3. response 是什么？

response 是服务端返回给客户端的结果。

例如客户端发送输入数据，服务端返回预测结果。

## 4. GET 是什么？

GET 通常用于获取数据。

例如：

GET /ping

用于检查服务是否正常。

## 5. POST 是什么？

POST 通常用于提交数据。

AI 推理接口通常使用 POST，因为输入数据可能是数组、文本、图片等复杂内容。

例如：

POST /predict

客户端发送：

```json
{
  "input": [1, 2, 3]
}