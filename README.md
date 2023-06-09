# Federated Learning Investigation of optimal settings under different levels of IID data

### 1. Folder Structure

```python
Repository
 |-- FL_experiments_colab.ipynb	# Main code notebook
 |-- FL_summary_charts.ipynb	# For visualisation of summary results
 |    |-- input               # input csv data which feeds into FL_experiments_colab.ipynb
 |    |    |-- obesity.csv    # for the Health Care dataset examined, the other 2 datasets are loaded from packages
 |    |-- images              # for ReadMe file             
```

### 2. Overview

Advancements in machine learning and artificial intelligence have enabled increasingly personalised on-demand services, with huge amounts of personal data being collected in real time.
Consequently, such advancements pose considerations regarding both data privacy amongst the user base and learning efficiency for data scientists. Traditional machine learning algorithms are centralised in their approach. For example, all local data is sent to a common server which facilitates the model training centrally, before releasing a global update back to all devices. However, with more devices being connected, this raises the issue of privacy and security concerns because such devices or sensors may potentially be collecting highly sensitive information from the end user. Additionally, as more data is being generated from connected devices, it is increasingly expensive and inefficient to facilitate the communication and storage of all data generated throughout the network. 

A Federated Learning (FL) approach to machine learning has been recently introduced to address such concerns. A FL approach decentralises the model training such that participating devices all train on their own local models, and only share their respective model updates to a central server, as shown below.

![result5](/images/result-5.png)

In this project, we aimed to find training parameters that will reduce the number of global communication rounds in training the overall FL model, and to achieve the best model accuracy.

There parameters included:
- FL aggregator selection: FedAvg, FedProx, FedPer and SCAFFOLD
- Label quantity distribution to each device (IID)
- Maximum number of samples in each local model
- Number of communication rounds
- Number of local training epochs
- Training batch size
- Number of active devices to participate in training

![result4](/images/result-4.png)

### 3. Experimental Framework

![result3](/images/result-3.png)
![result2](/images/result-2.png)

### 4. Results

The main findings were:
1. Using fewer active devices provided better results. This might be explained by finding clearer signal from fewer rather than more devices.
2. More epochs tend to improve accuracy in high IID settings, whereas fewer epochs work better in a low IID setting.
3. Smaller batch size is more applicable for high IID settings, whereas higher batch size is more appropriate for low IID settings.

![result1](/images/result-1.png)

#### Limitations

1. Methodology for data quantity skew did not have the intended effect on EMD
2. Impact of clients which have different computation power (systems heterogeneity)
3. GPU limitations which restricted parameter values for experimentation

#### Future Directions

Testing different deep learning model types was out of scope for our project. A potential area for future work that is not covered fully in literature is whether the model type, for example CNN or MLP, and associated construction choices such as dropout rate, batch normalisation, number and size of hidden layers influences the accuracy across IID levels. In addition, it would be useful to investigate whether the best performing centralised model is the same or similar to the best performing in a federated setting.

An evaluation limitation we found through our project was a lack of standard benchmarks for MNIST and CIFAR-10 FL models in existing literature. Hence it was difficult to assess how our results compared to other parameter combinations that have been reported. Related to this, we see future research benefitting from focusing on a wider range of datasets with different size and label imbalance, as we attempted to do through including the Healthcare dataset in our analysis. This would enable more robust validation of findings and generalisable conclusions to be formed.
