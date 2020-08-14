<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

  - [](#)
- [1.Data Structure and Algorithms](#1data-structure-and-algorithms)
  - [目录](#%E7%9B%AE%E5%BD%95)
- [2.Questions](#2questions)
  - [目录](#%E7%9B%AE%E5%BD%95-1)
  - [数组](#%E6%95%B0%E7%BB%84)
- [Appendix](#appendix)
    - [1. 如何进行代码测试？](#1-%E5%A6%82%E4%BD%95%E8%BF%9B%E8%A1%8C%E4%BB%A3%E7%A0%81%E6%B5%8B%E8%AF%95)
    - [2.刷题笔记](#2%E5%88%B7%E9%A2%98%E7%AC%94%E8%AE%B0)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

### 

> **既然终要承受痛苦，那么尝试思考的痛总归比承受学习的苦更有意义。**

如何快速有效地提高自己的编程能力：「[The Key To Accelerating Your Coding Skills](http://blog.thefirehoseproject.com/posts/learn-to-code-and-be-self-reliant/)」

> **For the rest of your life, go outside your limits every single day**

## 1.Data Structure and Algorithms

### 目录



## 2.Questions

### 目录

### 数组

| #    |         Title          | Finished |
| ---- | :--------------------: | -------- |
| 001  |   数组与泛型动态数组   |          |
| 002  | 1000万整数中查找某个数 |          |
| 003  |       约瑟夫问题       |          |

## 3.Datasets

本次我们选择英雄联盟数据集进行LightGBM的场景体验。英雄联盟是2009年美国拳头游戏开发的MOBA竞技网游，在每局比赛中蓝队与红队在同一个地图进行作战，游戏的目标是破坏敌方队伍的防御塔，进而摧毁敌方的水晶枢纽，拿下比赛的胜利。

现在共有9881场英雄联盟韩服钻石段位以上的排位比赛数据，数据提供了在十分钟时的游戏状态，包括击杀数、死亡数、金币数量、经验值、等级……等信息。列blueWins是数据的标签，代表了本场比赛是否为蓝队获胜。

数据的各个特征描述如下：

| 特征名称                 | 特征意义         | 取值范围 |
| ------------------------ | ---------------- | -------- |
| WardsPlaced              | 插眼数量         | 整数     |
| WardsDestroyed           | 拆眼数量         | 整数     |
| FirstBlood               | 是否获得首次击杀 | 整数     |
| Kills                    | 击杀英雄数量     | 整数     |
| Deaths                   | 死亡数量         | 整数     |
| Assists                  | 助攻数量         | 整数     |
| EliteMonsters            | 击杀大型野怪数量 | 整数     |
| Dragons                  | 击杀史诗野怪数量 | 整数     |
| Heralds                  | 击杀峡谷先锋数量 | 整数     |
| TowersDestroyed          | 推塔数量         | 整数     |
| TotalGold                | 总经济           | 整数     |
| AvgLevel                 | 平均英雄等级     | 浮点数   |
| TotalExperience          | 英雄总经验       | 整数     |
| TotalMinionsKilled       | 英雄补兵数量     | 整数     |
| TotalJungleMinionsKilled | 英雄击杀野怪数量 | 整数     |
| GoldDiff                 | 经济差距         | 整数     |
| ExperienceDiff           | 经验差距         | 整数     |
| CSPerMin                 | 分均补刀         | 浮点数   |
| GoldPerMin               | 分均经济         | 浮点数   |

## Appendix

#### 1. 如何进行代码测试？

- 

#### 2.刷题笔记

