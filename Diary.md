# First implementation

- use pybrain
- Song -> FFT -> NN -> IFFT -> Song and Song -> NN -> Song
- Feedforward network
- Will learn identity function (only one sample!)
- Since the identity function only needs the bias + output weights: number of hidden neurons does not matter

# Second try

- same setup as above but use sample that has artificial noise
- still yields dissatisfactory results on all hidden neuron counts
- Problems: PyBrain very limiting on error function and network structure

# Third try

- Recurrent in time domain
- Output: N samles at time t..t+N
- Input: Window of N samples from t..t+N
- To test: input window size, output one value

# Ideas

- Listen to a training algorithm:
  - train for duration of the song
  - then generate the next playback using the trained network
  - repeat

- Use 2D points for autoencoder not whole samples as vector
- Generate souds from Deep believe networks https://arxiv.org/pdf/1312.6034.pdf
- How to represent the input?
- Needs lots of musical pieces
- Can this be recurrent?
- We want to generate stuff, what is there to use?
- Trains network to reproduce one musical piece and put in another
