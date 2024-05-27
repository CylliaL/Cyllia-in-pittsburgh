# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:23:41 2024

@author: Xiang Li
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


## Load the data
path = 'E:/Pitts/ECON 2824 Big Data'
df = pd.read_csv(path+'/team_stats_2003_2023.csv')

## describe the data
df.info()

## Plot the data 
top_teams = df.groupby('team')['wins'].sum().nlargest(5).index
filtered_df = df[df['team'].isin(top_teams)]
filtered_df.groupby(['year', 'team']).sum()['wins'].unstack().plot(kind='area', stacked=True, figsize=(10, 6))
plt.title('Wins per Team Over the Years (Top 5 Teams)')
plt.xlabel('Year')
plt.ylabel('Wins')
plt.show()

"""
##Dictionary for myself because I know nothing about football/super bowl

points_opp:Points Against 
### 是指在体育比赛中，一个团队或球队所面对的对手得分总和。
### 在橄榄球中，如果一支球队在一个赛季中的"Points Against" 很低，说明他们的防守在比赛中相对成功

### points_diff: Point Differential
### 是指一个团队或球队在比赛中的得分差异，即球队的得分减去对手的得分。越高越好。
### Point Differential = Team’s Points Scored − Points Scored Against the Team

### mov: Average Margin of Victory
### 是指一个团队或球队在比赛中取得胜利时的平均得分优势。
### Average Margin of Victory = Total Points Differential in Wins/ Number of Wins

### total_yards: Offensive Yards Gained
### 通过传球（Passing Yards）和冲刺（Rushing Yards）等方式所获得的总码数，越高越好

### plays_offense: Offensive Plays Ran
### 比赛中，球队执行的进攻性战术或进攻性玩法的总次数。通常与其他进攻性统计数据（如进攻码数、得分等）一起使用

### yds_per_play_offense: Yards Per Play Offense
### 是指球队每次进攻性战术或进攻性玩法平均获得的码数。越高意味着球队每次进攻都能够获得更多的码数，显示出他们的进攻效率较高。
### Yards Per Play Offense = Total Offensive Yards/Total Offensive Plays Ran

### turnovers: Team Turnovers Lost
### 表示球队失去的总失误次数，其中失误包括例如传球被拦截Interceptions、失去橄榄球Fumbles Lost等导致对手取得球权的情况

### fumbles_lost: Team Fumbles Lost
### 表示球队在比赛中失去的橄榄球次数。球队的球员在比赛中持球时，由于一些原因失去橄榄球并被对方球队恢复。
### 失去橄榄球可能是一种失误，因为它导致了对手获得球权。对于评估球队在比赛中保护橄榄球的能力非常重要。

### first_down: First Downs Gained
### 表示球队在比赛中获得的总一阵进攻成功次数
### 球队需要在规定的四次进攻尝试内移动足够的码数，从而获得新的一阵进攻机会（即 First Down）

### ties: Ties
### 平局

#################################################################
"""

## Define the lable

x = df[['points', 'points_opp', 'total_yards', 'points_diff','win_loss_perc']]
y = df['wins']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)



## Ridge
ridge = Ridge()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

alphas = 10**np.linspace(10,-2,100)*0.5
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(x_train_scaled, y_train)

best_alpha = ridge_cv.alpha_

predr = ridge_cv.predict(x_test_scaled)  

## Calculate the MSE,MAS,R^2
mser = mean_squared_error(y_test, predr)
maer = mean_absolute_error(y_test, predr)
r2r = r2_score(y_test, predr)
print(f"Mean Squared Error: {mser}")
print(f"Mean Absolute Error (MAE): {maer}")
print(f"R^2 Score: {r2r}")

## Lasso
lasso = Lasso()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

alpha = 10**np.linspace(10,-2,100)*0.5
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(x_train_scaled, y_train)

best_alpha = lasso_cv.alpha_

lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(x_train_scaled, y_train)

predl = lasso_best.predict(x_test_scaled)  

## Calculate the MSE
msel = mean_squared_error(y_test, predl)
mael = mean_absolute_error(y_test, predl)
r2l = r2_score(y_test, predl)
print(f"Mean Squared Error: {msel}")
print(f"Mean Absolute Error (MAE): {mael}")
print(f"R^2 Score: {r2l}")

## Compare the two different models, Ridge has higher MAE but lower MSE and R^2 closer to 1
## So I chose Ridge Model

df = df.dropna(subset=['points', 'points_opp', 'total_yards', 'points_diff','win_loss_perc'])
chiefs_dta = df[(df['team'] == 'Kansas City Chiefs')]
san_dta = df[(df['team'] == 'San Francisco 49ers')]

features = pd.DataFrame({
    'points': (chiefs_dta['points'].values + san_dta['points'].values) / 2,
    'points_opp': (chiefs_dta['points_opp'].values + san_dta['points_opp'].values) / 2,
    'total_yards': (chiefs_dta['total_yards'].values + san_dta['total_yards'].values) / 2,
    'points_diff': (chiefs_dta['points_diff'].values + san_dta['points_diff'].values) / 2,
    'win_loss_perc': (chiefs_dta['win_loss_perc'].values + san_dta['win_loss_perc'].values) / 2,
    
})

print(chiefs_dta)
print(san_dta)
print(features)

## Predict

predicted_model = Ridge()
predicted_model.fit(x_train, y_train)

prediction = predicted_model.predict(features) 


prediction = prediction[0] if len(prediction) == 1 else prediction
average_prediction = np.mean(prediction)

print(average_prediction)






