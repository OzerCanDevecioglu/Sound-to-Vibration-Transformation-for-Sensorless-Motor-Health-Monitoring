# Sound-to-Vibration-Transformation-for-Sensorless-Motor-Health-Monitoring

# Project Description
![image](https://github.com/OzerCanDevecioglu/Sound-to-Vibration-Transformation-for-Sensorless-Motor-Health-Monitoring/assets/98646583/a2f1be95-2adc-4f77-86bb-deba397aa591)

Automatic sensor-based detection of motor failures such as bearing faults is crucial for predictive maintenance in various industries. Numerous methodologies have been developed over the years to detect bearing faults. Despite the appearance of numerous different approaches for diagnosing faults in motors have been proposed, vibration-based methods have become the de facto standard and the most commonly used techniques. However, acquiring reliable vibration signals, especially from rotating machinery, can sometimes be infeasibly difficult due to challenging installation and operational conditions (e.g., variations on accelerometer locations on the motor body), which will not only alter the signal patterns significantly but may also induce severe artifacts. Moreover, sensors are costly and require periodic maintenance to sustain a reliable signal acquisition. To address these drawbacks and void the need for vibration sensors, in this study, we propose a novel sound-to-vibration transformation method that can synthesize realistic vibration signals directly from the sound measurements regardless of the working conditions, fault type, and fault severity. As a result, using this transformation, the data acquired by a simple sound recorder, e.g., a mobile phone, can be transformed into the vibration signal, which can then be used for fault detection by a pre-trained model. The proposed method is extensively evaluated over the benchmark Qatar University Dual-Machine Bearing Fault Benchmark dataset (QU-DMBF), which encapsulates sound and vibration data from two different machines operating under various conditions. Experimental results show that this novel approach can synthesize such realistic vibration signals that can directly be used for reliable and highly accurate motor health monitoring.
[Paper Link](https://arxiv.org/abs/2305.07960)


## Qatar University Dual-Machine Bearing Fault Benchmark Dataset: QU-DMBF

![image](https://user-images.githubusercontent.com/98646583/207285515-23333c67-e1fe-41f3-a339-d39a3cfaeb68.png)


The benchmark dataset utilized in this study was established by Qatar University researchers using 2 different electric machines (Machine A and Machine B).  The experimental setup is given in Figure 1 which illustrates the orientation of the sensors and the installation of two machines. The configuration for Machine-A consists of a 3-phase AC motor, two double-row bearings, and a shaft rotating at a maximum speed 2840 RPM. A spring mechanism placed a 6 kips radial load on the shaft and bearing. PCB accelerometers (352C33high sensitivity Quartz ICP) mounted on the bearing housing. It weighs 180 kg and is 100x100x40cm. The working conditions for Machine-A are based on the following:  
•	19 different bearing configurations: 1 healthy, 18 fault cases: 9 with a defect on the outer ring, and 9 with a defect on the inner ring. The defect sizes vary from 0.35mm to 2.35mm.
•	5 different accelerometer localization: 3 different positions and 2 different directions (radial and axial)
•	2 different load (force) levels: 0.12 kN and 0.20 kN. 
•	3 different speeds: 480, 680, and 1010 RPM. 
We collected data for 270 seconds for each operating circumstance for a healthy bearing, and for 30 seconds for each faulty bearing case.  This results in a total time of 30 x 18 x 5 x 2 x 3 = 16,200 seconds of data measurement. The sound was also simultaneously recorded with the same sampling frequency as the vibration data.
In contrast, Machine B's design consists of a DC motor, two single-row bearings, and a shaft with a constant rotating speed of 2000 RPM. A spring mechanism installed a 6 kips radial load on the shaft and bearing. PCB accelerometers (353B33 high sensitivity Quartz ICP) mounted on the bearing housing. It weighs 3.5 kg, and the configuration measures 165x82x63 cm.  The working conditions for Machine B vary as follows:
•	19 different bearing configurations: 1 healthy, 9 with a defect on the outer ring, and 9 with a defect on the inner ring. The defect sizes vary from 0.35mm to 2.35mm.
•	6 different accelerometer positions.
•	A fixed load (force) of 0.40 kN. 
•	5 different speeds: 240, 360, 480, 700, and 1020 RPM. 
270 seconds of vibration/sound data for each operating condition for a healthy bearing are available in this dataset. As a result, the total time of the healthy bearing vibration data is 270 x 6 x 1 x 5 = 8,100 seconds. 30 seconds of vibration/sound data for each working condition for each faulty bearing are available.  This results in a 2:1 ratio of the faulty to healthy data, with a total time of 30 x 18 x 6 x 1 x 5 = 16,200 seconds. As a result, the dataset for machine B lasts 24,300 seconds in total (6.75 hours). The sound of each machine was simultaneously recorded with the same sampling frequency as the vibration data. The QU-DMBF is publicly shared in [Repo](https://github.com/OzerCanDevecioglu/Zero-Shot-Bearing-Fault-Detection-by-Blind-Domain-Transition) to serve as the dual-machine bearing fault detection benchmark.
As opposed to the challenges of vibration data collection, there is a crucial advantage for the sound signal acquisition as such a location sensitivity does not exist. This has been confirmed in a [Recent Study](https://github.com/OzerCanDevecioglu/Zero-Shot-Bearing-Fault-Detection-by-Blind-Domain-Transition) where even a DL classifier trained on the data acquired by one sensor may fail to detect certain faults in another’s data. The same study has further shown that the most reliable vibration data for fault detection is acquired from the closest accelerometer to the bearings, i.e., accelerometer-1 for both machines. So, we have selected this accelerometer for training the transformers of both machines and used them to synthesize the corresponding vibration signal, which is then evaluated with the actual vibration signal.

- Full QU-DMBF dataset with user manual can be downloaded from the given [link].
## Run

#### Train

![image](https://github.com/OzerCanDevecioglu/Sound-to-Vibration-Transformation-for-Sensorless-Motor-Health-Monitoring/assets/98646583/515a71de-048d-41cb-80b0-98a9dc1e03de)

- Training/Validation/Test dataset for two motors can be downloaded from the given [link](). Download train, validation and test data to the "tmats/", "vmats/", and "temats/" folders folders respectively. 
- Start training (Stage-1)
```http
  python Op_GAN_train.py
```

## Prerequisites
- Pyton 3
- Pytorch
- [FastONN](https://github.com/junaidmalik09/fastonn) 

## Results

![image](https://github.com/OzerCanDevecioglu/Sound-to-Vibration-Transformation-for-Sensorless-Motor-Health-Monitoring/assets/98646583/303098a3-c283-4566-bf0d-9311d190c490)

## Citation
If you find this project useful, we would be grateful if you cite this paper：

```http
O. C. Devecioglu, S. Kiranyaz, A. Elhmes, S. Sassi, T. Ince, O. Avci, M. H. Soleimani-Babakamali, E. Taciroglu and M. Gabbouj, “Zero-Shot Motor Health Monitoring by Blind Domain Transition ”
```


