# End-to-End Auditory Object Recognition via Inception Nucleus
Keras (Tensorflow) implementation of the paper: 

<div align="center">
  <b>Inception Block</b><br>
  <img src="figures/inception.png" width="500"><br><br>
</div>

## Notes:
- Proposing novel inception blocks for analyzing raw wave audio files. 
- We noticed the early layers are learning wavelet-like filters.
- Our analysis revealed that the network is learning semantically meaningful representations in the last layer.
- Using Global Average Pooling is very helpful in avoiding overfitting and reducing the number of parameters.


## How to re-run the experiments?

- Dataset can be downloaded here: http://urbansounddataset.weebly.com/urbansound8k.html
- After downloading the dataset you may extract it in "ds" folder.

```bash
git clone https://github.com/mkebrahimpour/e2e-Inception_Nucleus.git
cd e2e-Inception_Nucleus
sudo pip install -r requirements.txt
./run_all.sh # will run Inception Nucleus
```
## Representatoins Analysis
<div align="center">
  <b>First layer representatoins:</b><br>
  <img src="figures/filter0.png" width="200">
  <img src="figures/filter1.png" width="200">
  <img src="figures/filter2.png" width="200"><br>
  <img src="figures/filter3.png" width="200">
  <img src="figures/filter4.png" width="200">
  <img src="figures/filter5.png" width="200"><br>
</div>



<div align="center">
  <b> t-SNE on the last convolutional layer</b><br>
  <img src="figures/tsne2.png" width="600"><br><br>
</div>

<div align="center">
  <b>Inception Nucleus model - best accuracy: 88.4, trainable params = 289K</b><br>
  <img src="figures/icassp_loss.png" width="400">
   <img src="figures/icassp_accuracy.png" width="400">
</div>

