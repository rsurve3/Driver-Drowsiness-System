# Driver-Drowsiness-System

**Introduction**

In recent years, driver drowsiness has been one of the major causes of road accidents.
Every year of the number of fatalities due to drowsy drivers is increasing.
Statistics indicate the need of a reliable driver drowsiness detection system.
The system should be able to alert the driver to avoid any mishap.

**Objective**

Drowsiness Detection helps in detecting whether a person is feeling Drowsy or not with high accuracy.
Driver Drowsiness Detection will help to avoid accidents and save driver’s life.
Integration of the system in vehicles is expected to enhance driver’s performance and prevent accidents.

**Problem Statement**

Drowsy and fatigue driving is a major transportation safety concern and is responsible for thousands of accidents and numerous fatalities every year.
The resulting harms of drowsy/fatigue driving could be even higher among commercial vehicles.
The most important challenge is to detect the driver’s condition sufficiently early, prior to the onset of sleep, to avoid collisions. 

**System Implementation**

1. Acquire and Pre-process Dataset.
2. Train the CNN Model with available Dataset.
3. Save the features/checkpoint of CNN Model.
4. st the result of CNN Model with few static Images.

![image](https://user-images.githubusercontent.com/122759737/214500683-4e079aa8-9769-4161-be84-505cc80ae792.png)


5. Capture the Video through Webcam.
6. Extract the ROI from the Frames.
7. Feed the Images to trained Model.
8. Detect Drowsiness based on prediction of Model.
9. Alert the Driver if prediction is “Drowsy”.

![image](https://user-images.githubusercontent.com/122759737/214500701-10fa81b2-2e58-4764-b890-fe10acca670c.png)


**CNN Model**

CNN stands for Convolutional neural networks..
CNN models are the most popular and ubiquitous in the image data space.
The basic idea about how CNN will be used in our project is as follows:-
![image](https://user-images.githubusercontent.com/122759737/214500663-b32a905b-bd40-4bc2-b34b-12c5763bd14c.png)

**Flowchart**

![image](https://user-images.githubusercontent.com/122759737/214500732-5db10e47-bd3d-4501-9b8d-af6e733d5fae.png)


**Key Features of the System**

1. Dataset (4000+ Images)
2. Model predicts Drowsy and Non Drowsy (Images).
3. Eye Detection.
4. Detect Open Eyes.
5. Detect Close Eyes
6. Detect Drowsiness.
7. Alert Driver by Sound.

**Testing & Results**

**Part 1**

![image](https://user-images.githubusercontent.com/122759737/214500939-c173976b-8e34-4d81-84c9-6e2bf23bf6ca.png)

**Part 2**

![image](https://user-images.githubusercontent.com/122759737/214500968-8b9ad4fa-f774-4d24-9325-cb47fd67d900.png)

**Conclusion**

After implementation of the system, the model predicts accurately the frames which are “Drowsy” and which are “Non Drowsy”.
The overall system detects drowsiness with high accuracy and alerts the driver according to the results after detecting the Drowsy nature of driver.
Driver's Drowsiness Detection will help us to save lives by alerting the driver at initial state of “Drowsy” behaviour.

**Future Scope**

Along with Driver Drowsiness Detection using eyes we can also include others that attributes which can predict that an individual is Drowsy or not like yawn detection.
Also we can keep a track of head movement of the driver if the head is not straight or not focusing on the road the driver can be given some different type of alerts.
The future works may focus on the utilization of outer factors such as vehicle state, sleeping hours and weather conditions for fatigue measurement. 






