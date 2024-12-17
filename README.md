# CasMST-KAN
> **CasMST-KAN: A Cascade Prediction Model with Multi-Scale Temporal Convolution and Kernel Attention Networks**



---

## **Note**
 Some critical components of the code have been omitted to prevent premature disclosure before the official publication. Full code will be released upon paper acceptance/publication.

---

## **Getting Started**

### **Requirements**

The code has been tested with the following setup:
- **Python**: 3.9.13  
- **PyTorch**: 2.3.1  
- **CUDA Toolkit**: 12.4  
- **cuDNN**: 8.9.7  

Use [Anaconda](https://www.anaconda.com/) to set up the environment:

```bash
# Create a virtual environment
conda create --name CasMSTKan python=3.9 

# Activate the environment
conda activate CasMSTKan

# Install PyTorch and dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install additional requirements
pip install -r requirements.txt
```

---

## **Usage**

### **Data Preprocessing**
Navigate to the `preprocessing` directory and prepare the data:

```bash
cd ./preprocessing

# Preprocess the data and transform datasets to .pkl format
python utils.py
python preprocess_graph.py
```

You can customize the dataset, observation time, and other parameters in the `config.py` file.

### **Run the Model**

Run the CasMST-KAN model with the following commands:

```bash
# Navigate to the model directory
cd ./CasMSTKan_model

# Execute the main script
python run_CasMSTKan.py
```

---

## **Datasets**

See some sample cascades in `./data/twitter/dataset.txt`.

Weibo or Twitter Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) 

The datasets we used in the paper are come from:
- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).You can also download Twitter dataset [here](https://github.com/Xovee/casflow) in here.
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://github.com/CaoQi92/DeepHawkes) in here.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or feedback, please contact us. We welcome contributions to this repository!

