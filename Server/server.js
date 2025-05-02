// server.js
require('dotenv').config();
const express = require('express');
const axios   = require('axios');
const cors    = require('cors');
const OpenAI  = require('openai').default;

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 用來快取最新的分析結果
let lastResult = null;

app.use(cors());
app.use(express.json({ limit: '5mb' }));

// 接收 { imageUrl: string }
app.post('/analyze', async (req, res) => {
  try {
    const { imageUrl } = req.body;
    // 下載圖片
    const resp = await axios.get(imageUrl, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(resp.data, 'binary');

    // 呼叫 OpenAI Vision 模型
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: '你是影像顏色與材質分析專家。' },
        { role: 'user', content:
            '顏色選項：紅=01，黃=02，藍=03，綠=04，紫=05，橘=06，白=07，黑=08，咖啡色=09，灰=10；材質選項：絨毛=201，皮革=202，都不是=203。\n' +
            '請分析這張圖片，並回傳 JSON，格式如下：\n' +
            '{ "colors": ["01","03"], "material": "201" }。' }
      ],
      files: [
        { buffer, filename: 'img.jpg', mimetype: 'image/jpeg' }
      ]
    });

    const text = completion.choices[0].message.content;
    const result = JSON.parse(text);

    // 快取結果
    lastResult = result;

    res.json(result);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.toString() });
  }
});

// 新增：讓 Unity 或其他客戶端可以 GET 最新分析結果
app.get('/result', (req, res) => {
  if (!lastResult) {
    return res.status(404).json({ error: 'No analysis result available yet.' });
  }
  res.json(lastResult);
});

app.listen(process.env.PORT, () => {
  console.log(`分析伺服器跑在 http://localhost:${process.env.PORT}`);
});
