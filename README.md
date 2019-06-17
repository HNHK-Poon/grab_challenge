# Grab AI for S.E.A. (Based on telematics data, how might we detect if the driver is driving dangerously?)
![N|Solid](https://static.wixstatic.com/media/397bed_647eb21ff7ce4b3e990927d40781d7a7~mv2.png/v1/fill/w_490,h_302,al_c,usm_0.66_1.00_0.01/397bed_647eb21ff7ce4b3e990927d40781d7a7~mv2.png)

## Our solution
### ***Driving Safety Band***
 We trained a regression model to predict this safety band for multiple features based on the safe driving data (class is 0). In prediction, any values that exceed this safety band will trigger a warning that indicate possible of dangerous action. This can be implemented to Grab Apps by embedding video recording function. With this driving safety band, the algorithms will only select those scenes when abnormal behaviour detected. This can greatly reduce the load of the database and also protect the interest of both driver and passenger at the same time.

![Alt text](image/Safety%20band%20for%20Speed.png?raw=true "Title")

## Demo Video
[![](http://img.youtube.com/vi/rCUhDAE3VtM/0.jpg)](http://www.youtube.com/watch?v=rCUhDAE3VtM "Demo video")

## Introduction
This repository is to propose an idea of tackling driver safety issues with telematics data provided. Driver safety issue is a challenging problem in the world. Nowadays, we can leverage the power of big data and artificial intelligent on this problem. [Grab] is a company dedicated to provided safe and convinient transporation services to public. Today, Grab is present in eight countries across the region and become one of the largest transportation sharing platform in South East Asia.

## Problems
Dangerous driving is one of the main cause of deathly accidents. To ensure safety of drivers and passengers, we need to identify dangerous driving behaviours and educate drivers with safety driving behaviour. However, it's not easy to detect dangerous driving 
- **How to detect driver is driving dangerously based on telematics data?**

## Dataset
The dataset is a collection of driving's telematics data which cover around 20 thousands trips by Grab. The given dataset contains telematics data during trips (bookingID). Each trip will be assigned with label 1 or 0 in a separate label file to indicate dangerous driving. 

| Field | Description |
| ------ | ------ |
| bookingID | trip id |
| Accuracy | accuracy inferred by GPS in meters |
| Bearing | GPS bearing in degree |
| acceleration_x | accelerometer reading at x axis (m/s2) |
| acceleration_y | accelerometer reading at y axis (m/s2) |
| acceleration_z | accelerometer reading at z axis (m/s2) |
| gyro_x | gyroscope reading in x axis (rad/s)) |
| gyro_y | gyroscope reading in y axis (rad/s) |
| gyro_z | gyroscope reading in z axis (rad/s) |
| second | time of the record by number of seconds |
| Speed | speed measured by GPS in m/s |

### Limitation of data
Let's talk about some limitation of data provided for this challenge. As we know, these data is collected using sensors built-in driver smartphone. Also, the label (safe/dangerous) is given by the passengers without any guideline to determine a dangerous driving. This will caused problem of inconsistent data due to some of the major uncertainties as follow:
- **The way smartphone placed in car** (contribute to inconsistent of data)
- **Performance of the car** (contribute to inconsistent of data)
- **Concept of dangerous for passengers** (contribute to inconsistent of label)

### Summary of data analytic (will be further explain in notebook)
- Data is inconsistent and confusing for both safe and dangerous driving especially in columns of acceleration and gyro. No visible patterns can be used to distinguish safe or dangerous driving. Many of the safe driving data seems dangerous and many of dangerous driving data seems safe. 
- The result of deep learning classification using features given is not proomising.

# Proposed Solution: Safety Band
Based on analytic of data and some attempts to classify the trip, We've decided to focus on identify the possible dangerous moments in a trip instead of classify safety or per trip. This idea come from Bollinger Band in the stock market. 

![N|Solid](https://a.c-dn.net/c/content/igcom/en_GB/ig-financial-markets/market-news-and-analysis/trading-strategies/2018/11/27/trading-with-bollinger-bands/jcr%3Acontent/newspar3/panel_child_604225165/mp-parsys2/textimage/image.webimg.png/1543324397375.png)

**Bollinger band** is a famous technical analysis tools used in stock trading that provide relative definitions of high and low that can be used to create rigorous trading approaches. Here, we apply the idea of bollinger band with machine learning to detect any abnormal driving behaviours. We trained a regression model to predict this safety band for multiple features based on the safe driving data (class is 0). In prediction, any values that exceed this safety band will trigger a warning that indicate possible of dangerous action.

The proposed idea has advantages as follow:
- Can igonore the uncertainties (The way smartphone placed in car, Performance of the car, Concept of dangerous for passengers)
- Fair for both drivers and passengers, avoid unnecessary argues (No blackbox operation, can trace back the dangerous moments)
- No time restriction as trip come with various duration and deep learning model are mostly require fixed dimensions of inputs

## Usage

### Train
```sh
python train.py --file data/sampled_85.csv
```

### Predict
```sh
python predict.py --file samples
```
