import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지



#==========================================================================================================
# 문제 1. 남녀의 체중 데이터 생성
# np.random.normal(loc, scale, size) : 평균(loc), 표준편차(scale), 데이터 개수(size)인 정규분포에서 난수 생성
#==========================================================================================================

# 남자 : 100명, 평균: 70kg, 표준편차: 8kg
num_male = 100
male_weights = np.random.normal(loc = 70, scale = 8, size = num_male)
male_weights += np.random.normal(0, 1.5, num_male) # 잡음 추가(완벽한 정규분포가 아니도록 함)
male_weights += 1.0                                # bias 추가 (측정 오차나 체중계 오차를 흉내냄)

# 여자 : 300명, 평균 55kg, 표준편차: 7kg
num_female = 300
female_weights = np.random.normal(loc = 55, scale = 7, size = num_female)
female_weights += np.random.normal(0, 1.5, num_female) # 잡음 추가(완벽한 정규분포가 아니도록 함)
female_weights -= 0.5                                  # bias 추가 (측정 오차나 체중계 오차를 흉내냄)


#==========================================================================================================
# 문제 2. 예측 모델 생성
#==========================================================================================================

# 남자/여자 체중 데이터를 하나의 배열로 합치기
X = np.concatenate([male_weights, female_weights])


# 정답 레이블 생성(남자: 1, 여자: 0), 이진 라벨 배열
y = np.array([1] * num_male + [0] * num_female)


# 학습/시험용 데이터셋 (3:1 비율) 구분
# 400개 데이터를 랜덤으로 섞은 다음 300개/100개로 자르기
# :는 배열을 자르는 슬라이싱
def train_test_split_manual(X, y, test_ratio = 0.25):
    idx = np.random.permutation(len(X))   # 0~399 인덱스를 랜덤으로 섞기
    num_test = int(len(X) * test_ratio)  # 시험용 개수 = 400 * 0.25 = 100 개
    test_idx = idx[:num_test]            # 앞 100개는 시험용
    train_idx = idx[num_test:]           # 나머지 300개는 학습용
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y)  

train_male = X_train[y_train == 1] 
train_female = X_train[y_train == 0]


# 학습 데이터에서 남자/여자 따로 분리(MLE/MAP 계산용)
def gaussian_pdf(x, mu, sigma):
    # 정규분포 확률밀도함수(continuous 변수가 특정 구간 내의 값을 가질 상대적 가능성을 나타내는 함수) 직접 구현 <-> 공식 사용
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coeff * exponent


def gaussian_pdf_vec(x_arr, mu, sigma):
    # 배열 입력용 버전
    #리스트 컴프리헨션을 통해 배열을 하나씩 꺼내서 gaussian_pdf 함수를 적용한 결과를 새로운 배열로 반환
    return np.array([gaussian_pdf(x, mu, sigma) for x in x_arr])


def find_boundary(func, low, high, steps = 100000):
    # 두 함수가 교차하는 점 탐색
    # linspace란 : low에서 high까지 steps 개수만큼 균등하게 나눈 배열 생성
    # xs는 low에서 high까지 steps 개수만큼 균등하게 잘게 쪼갠 x값들의 배열
    xs = np.linspace(low, high, steps)
    for i in range(len(xs) - 1):
        if func(xs[i]) * func(xs[i+1]) < 0:  
            return (xs[i] + xs[i+1]) / 2 
    return None


#===========================================================================================================
# 문제 3. 각 모델의 학습 정확도(학습데이터셋), 시험 정확도(시험데이터셋) 측정
#===========================================================================================================

def accuracy(y_true, y_pred):
    # 정확도 계산
    return np.mean(y_true == y_pred)


# MLE : 학습 데이터에서 평균/표준편차 추정
mu_m_mle = np.mean(train_male)
sigma_m_mle = np.std(train_male)
mu_f_mle = np.mean(train_female)
sigma_f_mle = np.std(train_female)


# MLE 결정경계 : p(x|남자) = p(x|여자) 인 x 탐색
def mle_diff(x):
    return (gaussian_pdf(x, mu_m_mle, sigma_m_mle)
            - gaussian_pdf(x, mu_f_mle, sigma_f_mle)    
            )

boundary_mle = find_boundary(mle_diff, 50, 80)

# MLE 예측 함수
def predict_mle(x):
    lk_male = gaussian_pdf_vec(x, mu_m_mle, sigma_m_mle)
    lk_female = gaussian_pdf_vec(x, mu_f_mle, sigma_f_mle)
    return (lk_male >= lk_female).astype(int)   

acc_train_mle = accuracy(predict_mle(X_train), y_train)
acc_test_mle = accuracy(predict_mle(X_test), y_test)   

print(f"MLE 결정경계: {boundary_mle:.2f} kg")
print(f"MLE 학습 정확도: {acc_train_mle*100:.1f}% / 시험 정확도: {acc_test_mle*100:.1f}% ")


#사전확률 = 학습 데이터 내 비율
prior_male = len(train_male) / len(X_train)
prior_female = len(train_female) / len(X_train)

# MAP 결정경계 : p(x|남자)*prior_male = p(x|여자)*prior_female 인 x 탐색
def map_diff(x):
    return (gaussian_pdf(x, mu_m_mle, sigma_m_mle) * prior_male
            - gaussian_pdf(x, mu_f_mle, sigma_f_mle) * prior_female)

boundary_map = find_boundary(map_diff, 50, 80)

# MAP 예측 함수
def predict_map(x):
    post_male = gaussian_pdf_vec(x, mu_m_mle, sigma_m_mle) * prior_male
    post_female = gaussian_pdf_vec(x, mu_f_mle, sigma_f_mle) * prior_female
    return (post_male >= post_female).astype(int)

acc_train_map = accuracy(predict_map(X_train), y_train)
acc_test_map = accuracy(predict_map(X_test), y_test)

print(f"MAP 결정경계: {boundary_map:.2f} kg")
print(f"MAP 학습 정확도: {acc_train_map*100:.1f}% / 시험 정확도: {acc_test_map*100:.1f}% ")


#===========================================================================================================
# 문제 4. 두 변수로 확장 (체중 + 선호색상 = red(1) / blue(0)) 
# 체중과 선호색상은 서로 독립이라고 가정
# 선호색상의 분포는 남녀가 각각 편중된 경향을 가정
#===========================================================================================================

p_red_male = 0.3
p_red_female = 0.7

#binomial(n,p,size) : n번 시행해서 성공확률이 p인 시행을 size개수만큼 반복해서 성공한 횟수를 반환하는 함수
color_male = np.random.binomial(1, p_red_male, num_male)
color_female = np.random.binomial(1, p_red_female, num_female)

# 체중 + 선호색상을 하나의 2차원 배열로 합치기
# X2[:,0] = 체중, X2[:,1] = 선호색상
# 2차원 배열에서 모든 행(:)의 0번째 열에 체중 데이터를 집어넣고
# 2차원 배열에서 모든 행(:)의 1번째 열에 선호색상 데이터를 집어넣겠다는 의미

#np.column_stack : 여러 개의 1차원 배열을 열 방향으로 쌓아서 2차원 배열을 만드는 기능
X2 = np.column_stack([X, np.concatenate([color_male, color_female])])


# 학습/시험용 데이터셋 분리 (동일하게 3:1 비율)
X2_train, X2_test, y2_train, y2_test = train_test_split_manual(X2, y)

# 학습 데이터에서 남자/여자 따로 분리
train2_male = X2_train[y2_train == 1]
train2_female = X2_train[y2_train == 0]

# 2D MLE 파라미터 추정
# 체중: 정규분포, 선호색상: 베르누이 분포
mu2_m = np.mean(train2_male[:, 0])   # 남자 체중 평균
sigma2_m = np.std(train2_male[:, 0])   # 남자 체중 표준편차
mu2_f = np.mean(train2_female[:, 0]) # 여자 체중 평균
sigma2_f = np.std(train2_female[:, 0]) # 여자 체중 표준편차

p_red_m_est = np.mean(train2_male[:, 1])   # 학습 데이터에서 남자 red 확률 추정
p_red_f_est = np.mean(train2_female[:, 1]) # 학습 데이터에서 여자 red 확률 추정

# 사전 확률
prior2_male = len(train2_male) / len(X2_train)
prior2_female = len(train2_female) / len(X2_train)  

# 독립 가정 -> 결합 확률 = 체중 확률 x 색상 확률
def joint_likelihood(weight, color, mu, sigma, p_red):
    weight_prob = gaussian_pdf(weight, mu, sigma)   # 체중 확률밀도
    color_prob = p_red if color == 1 else (1 - p_red)  # 색상 확률
    return weight_prob * color_prob

# 2D MAP 예측 함수
def predict_2d_map(X2d):
    preds = []
    for weight, color in X2d:
        post_m = joint_likelihood(weight, color, mu2_m, sigma2_m, p_red_m_est) * prior2_male
        post_f = joint_likelihood(weight, color, mu2_f, sigma2_f, p_red_f_est) * prior2_female
        preds.append(1 if post_m >= post_f else 0)
    return np.array(preds)

acc2_train = accuracy(predict_2d_map(X2_train), y2_train)
acc2_test = accuracy(predict_2d_map(X2_test), y2_test)  

print(f"2D MAP 학습 정확도: {acc2_train*100:.1f}% / 시험 정확도: {acc2_test*100:.1f}% ")

#===========================================================================================================
# 문제 5. Linear Discriminant 사용
#===========================================================================================================

# 체중 정규화, 데이터를 평균 0, 표준편차 1로 변환
w_mean = np.mean(X2_train[:,0])
w_std = np.std(X2_train[:,0])

X2_train_norm = X2_train.copy()
X2_test_norm = X2_test.copy()
X2_train_norm[:,0] = (X2_train[:,0] - w_mean) / w_std  #정규화
X2_test_norm[:,0] = (X2_test[:,0] - w_mean) / w_std    #정규화

# 격자 탐색: w1, w2, b 조합을 모두 시도해서 가장 정확도 높은 파라미터 찾기
best_acc = 0
best_params = (0,0,0)

for w1 in np.arange(-2.0, 2.0, 0.1):
    for w2 in np.arange(-2.0, 2.0, 0.1):
        for b in np.arange(-2.0, 2.0, 0.1):
            score = X2_train_norm[:, 0] * w1 + X2_train_norm[:, 1] * w2 + b
            pred  = (score >= 0).astype(int)
            acc   = accuracy(pred, y2_train)
            if acc > best_acc:
                best_acc    = acc
                best_params = (w1, w2, b)

w1_best, w2_best, b_best = best_params
print(f"최적 파라미터: w1={w1_best:.2f}, w2={w2_best:.2f}, b={b_best:.2f}")

# 시험 데이터 정확도
score_test = X2_test_norm[:, 0] * w1_best + X2_test_norm[:, 1] * w2_best + b_best
pred_test  = (score_test >= 0).astype(int)
acc_ld_train = best_acc
acc_ld_test  = accuracy(pred_test, y2_test)

print(f"LD 학습 정확도: {acc_ld_train*100:.1f}% / 시험 정확도: {acc_ld_test*100:.1f}%")

# 결정경계 직선 시각화
# w1*체중_norm + w2*색상 + b = 0
# → 색상 = -(w1*체중_norm + b) / w2
w_norm_range = np.linspace(-3, 3, 200)
if abs(w2_best) > 1e-6:
    color_boundary = -(w1_best * w_norm_range + b_best) / w2_best

w_range_kg = w_norm_range * w_std + w_mean  # 원래 kg 단위로 복원

plt.figure(figsize=(8, 5))

m_idx = y2_train == 1
f_idx = y2_train == 0

plt.scatter(X2_train[m_idx, 0], X2_train[m_idx, 1] + np.random.uniform(-0.05, 0.05, m_idx.sum()),
            c='steelblue', alpha=0.4, label='남자', s=20)
plt.scatter(X2_train[f_idx, 0], X2_train[f_idx, 1] + np.random.uniform(-0.05, 0.05, f_idx.sum()),
            c='salmon', alpha=0.4, label='여자', s=20)
plt.plot(w_range_kg, color_boundary, 'k-', linewidth=2,
         label=f'결정경계: w1={w1_best:.2f}, w2={w2_best:.2f}, b={b_best:.2f}')

plt.xlabel('체중 (kg)')
plt.ylabel('선호색상 (1=Red, 0=Blue)')
plt.yticks([0, 1], ['Blue(0)', 'Red(1)'])
plt.ylim(-0.5, 1.5)
plt.title('Linear Discriminant 결정경계')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('plot_linear_discriminant.png', dpi=120)
plt.show()
