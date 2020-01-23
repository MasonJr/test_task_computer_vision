# Test task for junior Data Science Engineer

Computer Vision

Solved problem from Kaggle platform: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).

The dataset was taken from the data tab [dataset](https://www.kaggle.com/c/airbus-ship-detection/data).

The whole process of solving a task in ```main_notebook.ipynb```

Model tested in file ```test_model.ipynb```

Script for creation submission file is in ```submission_creator.ipynb```

Submission file located in: ```csv_data/sample_submission_v2.csv```

## Installation

You can download this repository by running this command in your terminal:
```bash
git clone https://github.com/MasonJr/test_task_computer_vision.git
```

All requirements in file ```requirements.txt```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the needed packages:

```bash
pip install -r requirements.txt
```
I used recommended tools and methods to accomplish the task:
* Python
* Keras
* U-Net architecture
* Dice score

## Description

**The model has been trained in only three epochs to show its correct work and to get some result in the test sample.**

A water surface image segmentation model has been created.
Implementation language Python. It used the U-Net architecture based on the Keras library.

The data was downloaded, then further information was processed and retrieved. After that, the methods of decoding and data encoding are implemented. A Data Generator class was created that contains the method of creating batches and the method of manipulationg the data.
The filtered data was divided into two samples:
* training
* validation.

Data aggregator created. Callbacks have been added to monitor the learning process. Then all the blocks were tested for the correctness of their work.

The final step is to load the model from a file and start training.

The Model was builded in file ```model.ipynb```
