# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.


## Approach
![image](https://github.com/crabyg/CLIP-implement/assets/105999203/bdbe25bb-7b18-474e-9c83-493aa4b8d84a)

##

# Partial CLIP Implementation

## Introduction
이 프로젝트는 OpenAI에서 발표한 CLIP("Learning Transferable Visual Models From Natural Language Supervision") 논문 중 일부 기능을 구현한 것입니다.

## Differences from Official Implementation

- **Tokenizer**: 오피셜코드는 자체적으로 CLIP Tokenizer를 제작한것으로 보임. 추가적으로 vit뿐만 아니라 resnet 기반 pretrain 모델도 제공하는 것을 확인
- **Img encoder**: 오피셜 코드는 VisionTransformer 인코더를 사용하였음
- **Text encoder**: 공식 코드는 일반적인 transformer구조를 변형하여 사용한것으로 확인
- **Clip encoder**: 공식코드는 로깃 스케일로 self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 다음과 같은 값을 적용하였음.


## Result

### train
train loss는 꾸준히 감소하는 추세를 보이지만, valid loss는 진동만 할 뿐 수렴하지 않는 모습을 보였음. 아마도 데이터의 양과 질에 문제가 있기때문으로 보이고, train loss가 감소하는 모습을 보아서 실제 clip model 최적화엔 성공하지 못하였지만 코드 구현은 성공했다고 판단
### zeroshot prediction
test 데이터에서 batch를 4로하여 임의로 4개의 이미지와 4개의 텍스트를 불러온 뒤 1개의 이미지에 대하여 나머지 4개의 텍스트와 코사인 유사도를 비교하여 가장 높은 값을 선택하는 방식으로 zeroshot prediction을 진행
![image](https://github.com/crabyg/CLIP-implement/assets/105999203/a25c9963-0226-47a2-a20b-3f04efcc521b)

실제 정답은 'Two boys in front of a soda machine' 이나, 다른 텍스트에 대해 유사도가 가장 높은 경향을 보임. train loss만 감소하였기 때문에 당연한 결과
### Discussion
학습의 성공보단 구현에 초점을 두었기 때문에 실제 CLIP에 비해 적은 데이터와 작은 모델을 사용하였고, 더욱 다양한 실험을 해보지 않았기 때문에 기회가 된다면 제대로 도전해볼 예정
