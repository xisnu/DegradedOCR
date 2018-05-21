# OCR for Degraded Bangla Documents
A CNN BLSTM CTC based implementation of Bangla Degraded OCR line recognition. This work is accepted as a conference paper in [ICPR 2018, 24th International Conference on Pattern Recognition](http://www.icpr2018.org/)
## Requirements
This model is implemented using
1. Python 2.7 (maintain this version)
2. Tensorflow 1.6+
3. H5py
4. Pillow
5. Numpy
## Usage Instruction
* Run the function ```makeh5_from_dir()``` from **PrepareDataset.py**. This will create a HDF file against image folder as specified e.g. **Data/Sample/Train/Line_Images"**. Run this for **Train** and **Test** directory seperatley. You need 2 HDF files.
* A CNN BLSTM CTC based network is implemented as in Figure:
![Model][model]

[model]: https://github.com/xisnu/DegradedOCR/blob/master/inception-ctc.jpg "Architecture"
* The network is given in **Hybrid_Model_Degraded.py**. Run the ```main()``` method as specified in comments.
* In ```Predict``` mode the network will genrate a file (**Predicted.txt**) containing actual annotation and network predicted strings.
