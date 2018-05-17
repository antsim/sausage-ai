# sausage-ai

Repository of the materials for my Web Dev & Sausages, vol. 7 talk in May 17th 2018. Session title is "Deep Learning Lyrics" and is about machine generating lyrics based on existing lyrics with Keras and Tensorflow.

## Getting Started

Contains environment.yml-file to use with Anaconda "conda env create". Download Anaconda from https://www.anaconda.com/download/  

Contains file tensorflow-1.8.0-cp36-cp36m-win_amd64.whl which you can also use if you have a hard time finding a good build from Tensorflow (that supports all available CPU instructions)  

Start training process by running  
```
python main.py ../data/cannibalcorpse.txt 400
```
Once you have trained the model and are ready to generate more lyrics than during the training, run:  
```
python lyrgen.py cannibalcorpse.h5 ../data/cannibalcorpse.txt 1.2
```
Last parameter is the diversity (temperature). Higher value means more errors but maybe more "innovative" results.

## Built With

* [Keras](https://keras.io/) - Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* [TensorFlow](https://www.tensorflow.org/) - TensorFlow is an open source software library for high performance numerical computation


## Authors

* **Antti Simonen** - *Initial work* - [antsim](https://github.com/antsim)

See also the list of [contributors](https://github.com/antsim/sausage-ai/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used, especially Ivan Liljeqvist - [ivan-liljeqvist](https://github.com/ivan-liljeqvist)
* Gofore Hackathon team who originally played around with this.
