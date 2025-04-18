# Image-Captioning-with-Encoder-Decoder-Architecture-Using-Deep-Learning

# overview
This project implements an Image Captioning model using a CNN-RNN architecture in PyTorch. The model leverages the Inception v3 pre-trained model for feature extraction from images and an LSTM-based decoder to generate captions. The architecture follows an encoder-decoder structure where the encoder extracts image features, and the decoder generates captions based on these features.

# Methodology: Image Captioning Model

  ## 1. EncoderCNN:
Description: Utilizes the Inception v3 pre-trained model for feature extraction from images. The final fully connected layer is replaced with a new linear layer to output features of a specified size (embed_size).

Explanation:

The Inception v3 model is a deep convolutional neural network (CNN) pre-trained on the ImageNet dataset and is effective for image classification tasks.

In the EncoderCNN class, the Inception v3 model is initialized with pre-trained weights, and the final fully connected layer (inception.fc) is replaced with a new linear layer to match the desired feature size (embed_size).

Additionally, ReLU activation and a dropout layer (dropout=0.5) are applied for non-linearity and regularization.

## 2. DecoderRNN:
Description: Implements an LSTM-based decoder for generating captions from the image features extracted by the EncoderCNN.

Explanation:

The DecoderRNN consists of an embedding layer (nn.Embedding), an LSTM layer (nn.LSTM), and a linear layer (nn.Linear) for predicting the next word in the caption.

The embedding layer converts words into dense vectors of embed_size dimensions.

The LSTM processes embedded words and image features, generating hidden states that capture sequential context.

The linear layer uses the LSTM hidden states to predict the next word in the caption.

A dropout layer (dropout=0.5) is applied for regularization.

## 3. CNNtoRNN (Overall Model):
Description: The CNNtoRNN class integrates the EncoderCNN and DecoderRNN into a single model for end-to-end image captioning.

Explanation:

The CNNtoRNN class combines the EncoderCNN and DecoderRNN, forming a complete image captioning model.

It takes an image as input, extracts features using EncoderCNN, and generates captions using DecoderRNN.

The model learns to map images to captions during training using the encoder-decoder architecture.

During inference, the trained model generates captions for new images by leveraging the learned features and parameters.

# Installation and Setup

    pip install -r requirements.txt

Make sure you have the following libraries:

    torch (PyTorch)
    
    torchvision
    
    spaCy
    
    matplotlib
    
    PIL
    
    numpy
    
    nltk
    
    bleu (for evaluation metrics)
