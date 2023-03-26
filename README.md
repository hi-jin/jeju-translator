## :capital_abcd: Korean standard language :arrow_right: Jeju dialect translator

> [AI Hub](https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)에서 제공한 AI데이터를 이용한 한국지능정보사회진흥원의 사업결과입니다.

## Desc

This is "standard :arrow_right: jeju dialect" translator.  
I fine-tuned the [pko-t5 model](https://huggingface.co/paust/pko-t5-base) on Hugging Face using the [jeju dialect dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=121).

## Directory structure

:open_file_folder: app : mobile translator application  
:open_file_folder: backend : backend to provide inference API  
-- :open_file_folder: model : translator model  

## License

This project uses a deep learning model from the [Original Repository](https://github.com/paust-team/pko-t5), which is licensed under the [MIT License](https://github.com/paust-team/pko-t5/blob/main/LICENSE).

```
Copyright 2022 PAUST

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```