# -*- coding: utf-8 -*-
import numpy as np

try:
    from circuit_optimization.encodings.registry import get_permutation, inverse_permutation
except ImportError:  # Fallback when running from within circuit_optimization/
    from encodings.registry import get_permutation, inverse_permutation


def _normalize_indexing(indexing):
    if indexing == "indexing":
        return None
    return indexing

def amplitude_encoding(images, indexing=None):
    """
    Calculate amplitude encoding states from input images.
    
    images : (batch, m, n) array-like
        A batch of mxn-pixel images to be encoded.
    
    indexing : None
                 --> index order is left to right, top to bottom
               'snake'
                 --> index order is meandering left to right and then
                     back right to left when going from top to bottom
               'hierarchical'FRQI
                 --> first two qubits specify quadrand, next two
                     specify sub-qudrant, and so on ...
               'gray'
                 --> index order is left to right, top to bottom, but
                     it is not labeled by binary integers rather by
                     the Gray code, i.e., two neighboring pixels differ
                     only by a single bit flip
    
    """
    
    # size and batchsize of the images
    m, n = images.shape[-2:]
    states = np.reshape(images.copy(), (-1,m,n))
    batchsize = states.shape[0]
    
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(m, n, indexing)
    states = states.reshape(batchsize, m * n)[:, idx].reshape(batchsize, m, n)
    
    # normalize the states
    states = np.reshape(states, (-1, m*n))
    states /= np.linalg.norm(states, axis=1)[:,None]
    return np.squeeze(states)

def amplitude_decoding(states, shape=(32,32), indexing=None):
    """
    Retrieve images from amplitude encoding states.
    
    states : (batch, n) array-like
        A batch of amplitude encoding states with n amplitudes.
    
    shape : tuple
        The shape of the image to be retrieved.
        Must be n = prod(shape).
    
    indexing : None or str
        Same as the indexing used for encoding.
    
    """
    
    # size and batchsize of the states
    images = np.reshape(states.copy(), (-1, *shape))
    batchsize = images.shape[0]
    
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(shape[0], shape[1], indexing)
    inv = inverse_permutation(idx)
    images = images.reshape(batchsize, -1)[:, inv].reshape(batchsize, *shape)
    
    return np.squeeze(images)

def FRQI_encoding(images, indexing='indexing', enc_type='trig'):
    """
    Calculate FRQI encoding states from input images.
    
    images : (batch, m, n) array-like
        A batch of mxn-pixel images to be encoded.
    
    indexing : None
                 -> index order is left to right, top to bottom
               'snake'
                 -> index order is meandering left to right and then
                    back right to left when going from top to bottom
               'hierarchical'
                 -> first two qubits specify quadrand, next two
                    specify sub-qudrant, and so on ...
               'gray'
                 --> index order is left to right, top to bottom, but
                     it is not labeled by binary integers rather by
                     the Gray code, i.e., two neighboring pixels differ
                     only by a single bit flip
    
    enc_type : 'trig'
                 -> Usual color qubit state
                        cos(x pi/2) |0> + sin(x pi/2) |1>
               'sqrt'
                 -> Encode color qubit state as
                        sqrt(1-x) |0> + sqrt(x) |1>
    
    """
    
    # size and batchsize of the images
    m, n = images.shape[-2:]
    states = np.reshape(images.copy(), (-1,m,n))
    batchsize = states.shape[0]
    num_qubits = int(np.log2(m*n))
    
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(m, n, indexing)
    states = states.reshape(batchsize, m * n)[:, idx].reshape(batchsize, m, n)

    # encode color qubits
    states = encode_greyscale(states.reshape(-1),
                              type_=enc_type)
    
    # normalize the states
    states = (states/2**(num_qubits/2)).reshape(batchsize, *(2,)*(num_qubits+1))
    return states

def FRQI_decoding(states, shape=(32,32), indexing='hierarchical', enc_type='trig'):
    """
    Retrieve images from FRQI encoding states.
    
    states : (batch, n) array-like
        A batch of amplitude encoding states with n amplitudes.
    
    shape : tuple
        The shape of the image to be retrieved.
        Must be n = prod(shape).
    
    indexing : None or str
        Same as the indexing used for encoding.
    
    enc_type : str
        Same as the type used for encoding.
    
    """
    
    # size and batchsize of the states
    num_qubits = int(np.log2(np.prod(shape)))

    images = np.reshape(states.copy(), (-1, 2**(num_qubits+1)))
    batchsize = images.shape[0]
    
    # decode color qubits
    images = decode_greyscale(images*2**(num_qubits/2), type_=enc_type)
    images = images.reshape(batchsize, *shape)
    
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(shape[0], shape[1], indexing)
    inv = inverse_permutation(idx)
    images = images.reshape(batchsize, -1)[:, inv].reshape(batchsize, *shape)
    
    return np.squeeze(images)

def NEQR_encoding(images, q, indexing=None):
    """
    Calculate NEQR encoding states from input images.
    
    images : (batch, m, n) array-like
        A batch of mxn-pixel images to be encoded.
    
    q : int
        Use q color qubits in the encoding.
        The number of discrete color values is 2**q.
    
    indexing : None
                 -> index order is left to right, top to bottom
               'snake'
                 -> index order is meandering left to right and then
                    back right to left when going from top to bottom
               'hierarchical'
                 -> first two qubits specify quadrand, next two
                    specify sub-qudrant, and so on ...
               'gray'
                 --> index order is left to right, top to bottom, but
                     it is not labeled by binary integers rather by
                     the Gray code, i.e., two neighboring pixels differ
                     only by a single bit flip
    
    """
    
    # size and batchsize of the states
    m, n = images.shape[-2:]
    states = np.reshape(images.copy(), (-1,m,n))
    batchsize = states.shape[0]
    num_qubits = int(np.log2(m*n))
    
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(m, n, indexing)
    states = states.reshape(batchsize, m * n)[:, idx].reshape(batchsize, m, n)
    
    # encode color qubits
    states = encode_disc_greyscale(states, q)
    
    # normalize the states
    states = (states/2**(num_qubits/2)).reshape(batchsize, *(2,)*(num_qubits+q))
    return np.squeeze(states)

def NEQR_decoding(states, q, shape=(32,32), indexing=None):
    """
    Retrieve images from NEQR encoding states.
    
    states : (batch, n) array-like
        A batch of amplitude encoding states with n amplitudes.
    
    q : int
        Use q color qubits in the encoding.
        The number of discrete color values is 2**q.
    
    shape : tuple
        The shape of the image to be retrieved.
        Must be n = prod(shape).
    
    indexing : None or str
        Same as the indexing used for encoding.
    
    """
    
    # size and batchsize of the states
    num_qubits = int(np.log2(np.prod(shape)))
    images = np.reshape(states.copy(), (-1, 2**(num_qubits+q)))
    batchsize = images.shape[0]
    
    # decode color qubits
    images = decode_disc_greyscale(images*2**(num_qubits/2), q)
    images = images.reshape(batchsize, *shape)
    
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(shape[0], shape[1], indexing)
    inv = inverse_permutation(idx)
    images = images.reshape(batchsize, -1)[:, inv].reshape(batchsize, *shape)
    
    return np.squeeze(images)

def discretize_image(image, q):
    """
    Discretizes the continuous greyscale values of the image to 2**q different
    discrete greyscale values.
    
    image : array_like
        The original image with continuous greyscale values as pixel values.
    
    q : int
        The number of discrete greyscale values in [0, 1] is 2**q.
    
    """
    disc_img = (np.array(image)*2**q).astype(int)
    disc_img[disc_img == 2**q] = 2**q-1
    disc_img = disc_img.astype(float)/(2**q-1)
    return disc_img

def encode_greyscale(p, type_='trig'):
    """
    Encodes an array of pixel values into an array of quantum states
    encoding the pixel values.
    
    p : array_like
        An array of greyscale pixel values to be encoded in quantum states.
    
    type_ : str, optional
        The type of encoding to be used.
        'trig' : Final state is [cos(p*pi/2), sin(p*pi/2)]
        'sqrt' : Final state is [sqrt(p), sqrt(1-p)]
        The default is 'trig'.
    
    """
    p_in = np.reshape(p, (-1,))
    if type_ == 'trig':
        return np.array([np.cos(np.pi*p_in/2), np.sin(np.pi*p_in/2)]).T
    elif type_ == 'sqrt':
        return np.array([np.sqrt(1-p_in), np.sqrt(p_in)]).T

def decode_greyscale(state, type_='trig'):
    """
    Decodes the greyscale value encoded in a quantum state.
    
    state : array_like
        A list of quantum states to be decoded.
    
    type_ : str, optional
        The type of encoding that was used to encode the greyscale value.
        'trig' : Encoded as [cos(p*pi/2), sin(p*pi/2)]
        'sqrt' : Encoded as [sqrt(p), sqrt(1-p)]
        The default is 'trig'.
    
    """
    p = np.reshape(state, (-1,2))[:,0].real
    p[p<-1.] = -1.
    p[p> 1.] =  1.
    if type_ == 'trig':
        return (np.arccos(p)*2/np.pi)
    elif type_ == 'sqrt':
        return p**2

def encode_disc_greyscale(p, q):
    """
    Encodes an array of discrete pixel values into an array of quantum
    states encoding the pixel values.
    
    p : array_like
        An array of discrete greyscale pixel values to be encoded in
        quantum states.
    
    q : int
        The number of color qubits,which corresponds to 2**q discrete
        greyscale levels.
    
    """
    p_in = (np.reshape(p, -1)*(2**q-1)).astype(int) # turn p back into integers in {0, 1, ..., 2**n-1}
    state = np.zeros((p_in.shape[0], 2**q))
    state[range(p_in.shape[0]), p_in] = 1
    return state

def decode_disc_greyscale(state, q):
    """
    Decodes the discrete greyscale value encoded in a quantum state.
    
    state : array_like
        A list of quantum states to be decoded.
    
    q : int
        The number of color qubits,which corresponds to 2**q discrete
        greyscale levels.
    
    """
    inp_state = np.reshape(state, (-1, 2**q)) # reshape into index and state
    p = np.argmax(inp_state, axis=1)/(2**q-1)
    return p

def hierarchical_index(n):
    """
    Calculate the hierarchical indices of a nxn array.
    
    n : int
        The level of the hierachical index, corresponds
        to image size (2**n, 2**n).
    
    """
    
    Z_index = np.arange(4).reshape(2,2)
    index = Z_index
    for i in range(n-1):
        index = np.kron(index*4, np.ones((2,2), dtype=np.int32))\
            + np.kron(np.ones(index.shape, dtype=np.int32), Z_index)
    return index

def binary_to_gray(num):
    """
    Convert binary integer to Gray code bitstring.
    
    num : int or ndarray of type int
        The integer(s) to be converted to Gray code.
    
    """
    
    return num ^ (num >> 1)

def gray_to_binary(num):
    """
    Convert Gray code bitstring to binary integer.
    
    num : int or ndarray of type int
        The integer(s) to be converted from Gray code.
    
    """
    
    shift = 2**int(np.floor(np.log2(np.ceil(np.log2(np.max(num)+1)))))
    while shift > 0:
        num ^= num >> shift
        shift //= 2
    return num

def move_qubits_left(states, num=1):
    """
    Permute qubits cyclically to the left:
    (q_1, q_2, ..., q_n) -> (q_num+1, q_num+2, ..., q_num)
    
    states : ndarray
        The batch of n-qubit states where the qubits are shifted.
    
    num : int
        The number of qubits to shift by.
    
    """
    
    if len(states.shape) > 1:
        batchsize = states.shape[0]
    else:
        batchsize = 1
    states = states.reshape(batchsize, -1)
    L = int(np.log2(states.shape[1]))
    
    # bring state into right shape
    states = states.reshape(batchsize, *(2,)*L)
    # transpose
    states = states.transpose(0,*range(num+1,L+1),*range(1,num+1))
    
    return states.reshape(batchsize, -1)

def move_qubits_right(states, num=1):
    """
    Permute qubits cyclically to the right:
    (q_1, q_2, ..., q_n) -> (q_n-num+1, q_n-num+2, ..., q_n-num)
    
    states : ndarray
        The batch of n-qubit states where the qubits are shifted.
    
    num : int
        The number of qubits to shift by.
    
    """
    
    if len(states.shape) > 1:
        batchsize = states.shape[0]
    else:
        batchsize = 1
    states = states.reshape(batchsize, -1)
    L = int(np.log2(states.shape[1]))
    
    # bring state into right shape
    states = states.reshape(batchsize, *(2,)*L)
    # transpose
    states = states.transpose(0,*range(L-num+1,L+1),*range(1,L-num+1))
    
    return states.reshape(batchsize, -1)

def FRQI_RGBa_encoding(images, indexing='hierarchical'):
    """
    images : (batch, m, n, 3) ndarray
    
    indexing : None, 'hierarchical' or 'snake'
    
    """
    # images
    images = images.copy()
    batch, m, n, _ = images.shape
    indexing = _normalize_indexing(indexing)
    idx = get_permutation(m, n, indexing)
    images = images.reshape(batch, m * n, 3)[:, idx, :]
    # map pixels to states
    states = np.zeros((batch, 2**3, m*n))
    funcs = [lambda x: np.cos(np.pi * x/2), lambda x: np.sin(np.pi * x/2)]
    for i in range(3):
        states[:, i  , :] = funcs[0](images[:, :, i])
        states[:, i+4, :] = funcs[1](images[:, :, i])
    states[:, 3, :] = 1.
    # normalize states
    states = states.reshape(batch, 2**3*m*n)/np.sqrt(m*n)/2
    return states

def FRQI_RGBa_decoding(states, indexing='hierarchical', shape=(32,32)):
    """
    states : (batch, 2**3 * m * n) ndarray
    
    indexing : None, 'hierarchical' or 'snake'
    
    shape : tuple (m, n)
    
    """
    # states
    states = states.copy()
    if len(states.shape) > 1:
        batch = states.shape[0]
    else:
        batch = 1
    # map states to pixels
    channels = []
    states = states.reshape(batch, 2**3, -1)
    for i in range(3):
        channel = (states[:, i, :]**2 - states[:, i+4, :]**2) * states.shape[-1] * 4
        channel[channel > 1.] = 1.
        channel[channel <-1.] =-1.
        channels.append(np.arccos(channel)/np.pi)
    images = np.stack(channels, axis=-1).reshape(batch, -1, 3)

    indexing = _normalize_indexing(indexing)
    idx = get_permutation(shape[0], shape[1], indexing)
    inv = inverse_permutation(idx)
    images = images[:, inv, :].reshape(batch, *shape, 3)
    return images
