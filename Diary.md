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
