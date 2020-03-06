Directory of Sensor Project

This project focuses on the classification of faulty components using sensor data in manufacturing machinery. Feature engineering is the heart of 
the classic approach to signal processing and machine learning using features extracted from the output of algorithms such as the Fast Fourier 
Transform (FFT) and Discrete Wavelet Transform (DWT) for this paper. The contemporary approach will use 1D CNN and raw signals as input to the 
1D CNN model. There is no need to input features to the 1D CNN; the model will automatically define features (feature maps) through backpropogation. 
Both approaches will be used in this paper and the results compared. Finally, the classic and contemporary approaches will be combined by extracting 
features from 1D CNN, FFT and DWT and then conventional classification algorithms will be used.

CapstoneP2

ModelApproachSensoTimeSeriesData.pdf - powerpoint in pdf format which describes the two approaches, contemporary and classic approaches with results.

SensorProjectFinalResults.pdf - final paper with details on the project, data acquisition, EDA, analysis and results.


Code

	DiscreteWaveletTransformSignals.py - shows the functionality of the Discrete Wavelet Transform (DWT) using generated cosine data.

	DiscreteWaveletTransformSignals_BearingData.py - shows how feature extraction will be performed using DWT with bearing sensor data examples.

	FourierTransformSignals.py - shows the functionality of the Fast Fourier Transform (FFT) using generated cosine data.

	FourierTransformSignals_BearingData.py - shows how feature extraction will be performed using FFT with bearing sensor data examples.

	SignalProcessingDWTEngineeredFeatures.py - classification model using DWT extracted features for the model with sensor data.

	SignalProcessingFFTEngineeredFeatures.py - classification model using FFT extracted features for the model with sensor data.

	SignalProcessingFFTDWTEngineeredFeatures.py - classification model using FFT and DWT extracted features for the model with sensor data.

	SignalProcessingusing1DCNN.py - classificaton model using one dimensional convolutional neural networks with sensor data.

	SignalProcessingFFTDWT1DCNNEngineeredFeatures - classification model using 1D CNN, FFT and DWT engineered features.

Notebooks

	SignalAnalysisforFeatureEngineering.ipynb - Jupyter Notebook which shows the functionality of the Fast Fourier Transform (FFT) and Discrete Wavelet Transfrom (DWT).  

	SignalFeatureEngineeringforBearingData.ipynb - Jupyter Notebook describing the approach using FFT and DWT and feature engineering with examples using ball bearing sensor data.

	SignalMachineLearning_BearingData.ipynb - Jupyter Notebook containing all of the machine learning model executions using contemporary approach (1D CNN) and the classic approach using FFT and DWT feature extraction.

	img - directory containing all of the images for the above three notebooks.

ReportOuts

	ModelApproachSensoTimeSeriesData.pdf - powerpoint in pdf format which describes the two approaches, contemporary and classic approaches with results.

	SensorProjectFinalResults.pdf - final paper with details on the project, data acquisition, EDA, analysis and results.