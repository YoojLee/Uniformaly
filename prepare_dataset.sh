## Unzip the attached zip file (MVTecAD, BTAD, MTD, CPD, mvtec_all, SemanticAD)
### mvtec_all is for multi-class anomaly detection
# mkdir data

# unzip anomaly_detection.zip -d data
# unzip species_60.zip -d data

# Due to the limits of file size, we only provide the MVTec-AD dataset
mkdir mvtec
cd mvtec
# Download MVTec anomaly detection dataset
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz