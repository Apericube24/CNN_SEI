import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def ReLu(x):
    return np.maximum(0,x)

def softmax(x):
    sum = 0
    for i in range(len(x)):
        sum += np.exp(x[i])
    return np.exp(x) / sum

def MaxPool(M):
    y,x,num_channels = M.shape
    i = 0
    j = 0
    output_height = (y - 3) // 2 + 1
    output_width = (x - 3) // 2 + 1
    print(f"output height: {output_height}, output width: {output_width}")
    output = np.zeros((output_height, output_width, num_channels))

    for c in range(num_channels):
        for i in range(0, output_height):
            for j in range(0, output_width):
                start_i, start_j = i * 2, j * 2
                region = M[start_i:start_i + 3, start_j:start_j + 3, c] #j'ai enlever output x et y car pas besoin et re for pour eviter de remettre à 0
                output[i, j, c] = np.max(region)
    return output

def MaxPool_v2(M):
    y,x,num_channels = M.shape
    i = 0
    j = 0
    output_height = y // 2
    output_width = x // 2
    output = np.zeros((output_height, output_width, num_channels))
    for c in range(num_channels):
        for i in range(0, output_height):
            for j in range(0, output_width):
                start_i, start_j = i * 2, j * 2
                if (start_i + 3 > y):
                    if(start_j + 3 > x):
                        region = M[start_i:y, start_j:start_j + 3, c]
                    else:
                        region = M[start_i:y, start_j:x, c]
                elif (start_j + 3 > x):
                    region = M[start_i:start_i + 3, start_j:x, c]
                else:
                    region = M[start_i:start_i + 3, start_j:start_j + 3, c] #j'ai enlever output x et y car pas besoin et re for pour eviter de remettre à 0
                output[i, j, c] = np.max(region)
    return output

def convolution(image, Ks, biais):
    image_height, image_width, image_channels = image.shape
    K_height, K_width, _, num_filters = Ks.shape

    # Calcul des dimensions de sortie
    output_height = image_height
    output_width = image_width

    # Initialisation de la sortie
    output = np.zeros((output_height, output_width, num_filters))

    # Ajout de padding pour conserver les dimensions de l'image
    pad_h = K_height // 2  # Padding sur la hauteur
    pad_w = K_width // 2  # Padding sur la largeur
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    # Convolution
    for f in range(num_filters):  # Parcours des filtres
        conv_sum = np.zeros((output_height, output_width))  # Somme intermédiaire pour chaque filtre
        for c in range(image_channels):  # Parcours des canaux d'entrée
            for i in range(output_height):  # Parcours de la hauteur
                for j in range(output_width):  # Parcours de la largeur
                    # Extraction de la région locale (patch de l'image)
                    region = padded_image[i:i+K_height, j:j+K_width, c]
                    # Produit élément par élément entre le patch et le noyau
                    conv_sum[i, j] += np.sum(region * Ks[:, :, c, f])
        # Application du biais et de l'activation ReLU
        output[:, :, f] = ReLu(conv_sum + biais[f])

    return output

def c_reshape(M):
    height, width, channels = M.shape
    output = M
    return output.reshape(height*width*channels) # On garde comme ça pour l'instant mais à changer pour le matériel

def FCP(M, weights, bias): # Fully Connected Perceptron
    '''
    M : (180,)
    weights : (180, 10)
    bias : (10,)
    output : (10,)
    '''
    M = M.T
    output = softmax(M.dot(weights) + bias)
    return output


def normalize_image(I):
    N = I.size
    mu = np.mean(I)
    sigma = np.std(I)

    normalized = (I - mu) / max(sigma, 1 / np.sqrt(N))

    return normalized

def normalize_image_bis(image):
    """
    Normalise une image 32x32 sur ses composantes RGB sans utiliser numpy.
   
    Args:
        image (list): Image 3D (32x32x3) sous forme de liste de listes pour les pixels RGB.
                      Chaque pixel est une liste de trois valeurs entières (R, G, B).
   
    Returns:
        list: Image normalisée avec les mêmes dimensions.
    """
    height = len(image)
    width = len(image[0])
    channels = len(image[0][0])  # En supposant que c'est une image RGB
    N = height * width * channels

    # Calcul de la moyenne (µ)
    total = 0
    for row in image:
        for pixel in row:
            total += sum(pixel)  # Somme des trois canaux RGB
    mu = total / N
    # Calcul de l'écart-type (σ)
    variance = 0
    for row in image:
        for pixel in row:
            for value in pixel:
                variance += (value - mu) ** 2
    sigma = (variance / N) ** 0.5
    # Calcul de la normalisation
    normalized_image = []
    denominator = max(sigma, 1 / (N ** 0.5))
    for row in image:
        normalized_row = []
        for pixel in row:
            normalized_pixel = [(value - mu) / denominator for value in pixel]
            normalized_row.append(normalized_pixel)
        normalized_image.append(normalized_row)

    return np.array(normalized_image)


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8)
        
    data = data.reshape(-1, 3073)
    labels = data[:, 0]
    images = data[:, 1:]
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    resized_images = np.array([image[3:27, 3:27] for image in images])
    return resized_images, labels

def load_class_names(file_path):
    with open(file_path, 'r') as file:
        class_names = [line.strip() for line in file]
    return class_names


def plot_sample_image(images, labels, class_names, index=0):
    image = images[index]
    label = labels[index]
    class_name = class_names[label]
    print(image.shape)
    plt.imshow(image)
    plt.title(f"Label: {label} ({class_name})")
    plt.axis('off')
    plt.show()

images, labels = load_cifar10_batch('cifar-10-batches-bin/data_batch_1.bin')
class_names = load_class_names('cifar-10-batches-bin/batches.meta.txt')


def process_tensor_data(tensor_name, raw_data):
    cleaned_string = raw_data.replace('[', '').replace(']', '').replace('\n', ' ')
    float_list = [float(x) for x in cleaned_string.split()]
    tensor_dict[tensor_name] = np.array(float_list)

tensor_dict = {}

with open('CNN_coeff_3x3.txt', 'r') as file:
    tensor_name = None
    raw_data = ''
    
    for line in file:
        if line.startswith('tensor_name:'):
            if tensor_name:
                process_tensor_data(tensor_name, raw_data)
            tensor_name = line.split(':')[1].strip()
            raw_data = ''
        else:
            raw_data += line.strip() + '\n'
    
    if tensor_name:
        process_tensor_data(tensor_name, raw_data)

def reshape_array_parameters():
    for tensor_name, parameter_array in tensor_dict.items():
        if tensor_name == 'conv1/biases':
            tensor_dict[tensor_name] = parameter_array.reshape(64,)
        elif tensor_name == 'conv1/weights':
            tensor_dict[tensor_name] = parameter_array.reshape(3, 3, 3, 64)
        elif tensor_name == 'conv2/biases':
            tensor_dict[tensor_name] = parameter_array.reshape(32,)
        elif tensor_name == 'conv2/weights':
            tensor_dict[tensor_name] = parameter_array.reshape(3, 3, 64, 32)
        elif tensor_name == 'conv3/biases':
            tensor_dict[tensor_name] = parameter_array.reshape(20,)
        elif tensor_name == 'conv3/weights':
            tensor_dict[tensor_name] = parameter_array.reshape(3, 3, 32, 20)
        elif tensor_name == 'local3/biases':
            tensor_dict[tensor_name] = parameter_array.reshape(10,)
        elif tensor_name == 'local3/weights':
            tensor_dict[tensor_name] = parameter_array.reshape(180, 10)

reshape_array_parameters()

def plot_predict_image(images, labels, class_names, predict, index):
    image = images[index]
    label = labels[index]
    class_name = class_names[label]
    predict_class = class_names[predict]
    plt.imshow(image)
    plt.title(f"Label: {label} ({class_name})\n Predict: {predict} ({predict_class})")
    plt.axis('off')
    plt.show()

def predict(begin, end):
    i = begin
    correct = 0
    toto0=0
    toto1=0
    toto2=0
    toto3=0
    toto4=0
    toto5=0
    toto6=0
    toto7=0
    toto8=0
    toto9=0
    ctoto0=0
    ctoto1=0
    ctoto2=0
    ctoto3=0
    ctoto4=0
    ctoto5=0
    ctoto6=0
    ctoto7=0
    ctoto8=0
    ctoto9=0
    while (i <= end):
        input_image = images[i]
        input_image = normalize_image_bis(input_image)
        k1 = tensor_dict['conv1/weights']
        biais1 = tensor_dict['conv1/biases']
        k2 = tensor_dict['conv2/weights']
        biais2 = tensor_dict['conv2/biases']
        k3 = tensor_dict['conv3/weights']
        biais3 = tensor_dict['conv3/biases']
        weights_fcp = tensor_dict['local3/weights']
        biais_fcp = tensor_dict['local3/biases']

        conv1 = convolution(input_image, k1, biais1)
        maxpool1 = MaxPool_v2(conv1)
        conv2 = convolution(maxpool1, k2, biais2)
        maxpool2 = MaxPool_v2(conv2)
        conv3 = convolution(maxpool2, k3, biais3)
        maxpool3 = MaxPool_v2(conv3)
        reshape = c_reshape(maxpool3)
        output = FCP(reshape, weights_fcp, biais_fcp)
        index_max = np.argmax(output)
        #print(index_max)
        print(f"Loading : {i-begin}/{(end-begin)}")
        # plot_predict_image(images, labels, class_names, index_max, i)
        if index_max == labels[i]:
            correct = correct + 1
            if labels[i]==0 :
                ctoto0 = ctoto0 + 1
            if labels[i]==1 :
                ctoto1 = ctoto1 + 1
            if labels[i]==2 :
                ctoto2 = ctoto2 + 1
            if labels[i]==3 :
                ctoto3 = ctoto3 + 1
            if labels[i]==4 :
                ctoto4 = ctoto4 + 1
            if labels[i]==5 :
                ctoto5 = ctoto5 + 1
            if labels[i]==6 :
                ctoto6 = ctoto6 + 1
            if labels[i]==7 :
                ctoto7 = ctoto7 + 1
            if labels[i]==8 :
                ctoto8 = ctoto8 + 1
            if labels[i]==9 :
                ctoto9 = ctoto9 + 1     
        i = i + 1
        if labels[i]==0 :
            toto0 = toto0 + 1
        if labels[i]==1 :
            toto1 = toto1 + 1
        if labels[i]==2 :
            toto2 = toto2 + 1
        if labels[i]==3 :
            toto3 = toto3 + 1
        if labels[i]==4 :
            toto4 = toto4 + 1
        if labels[i]==5 :
            toto5 = toto5 + 1
        if labels[i]==6 :
            toto6 = toto6 + 1
        if labels[i]==7 :
            toto7 = toto7 + 1
        if labels[i]==8 :
            toto8 = toto8 + 1
        if labels[i]==9 :
            toto9 = toto9 + 1        
        
    reussite = correct / (end - begin)
    print(f"pourcentage reussite: {reussite*100}%")
    print("\n\n")
    print(f"pourcentage reussite de airplane : {ctoto0/toto0}% avec {toto0} ech \n")
    print(f"pourcentage reussite de automobile  : {ctoto1/toto1}% avec {toto1} ech \n")
    print(f"pourcentage reussite de bird : {ctoto2/toto2}% avec {toto2} ech \n")
    print(f"pourcentage reussite de cat : {ctoto3/toto3}% avec {toto3} ech \n")
    print(f"pourcentage reussite de deer : {ctoto4/toto4}% avec {toto4} ech \n")
    print(f"pourcentage reussite de dog : {ctoto5/toto5}% avec {toto5} ech \n")
    print(f"pourcentage reussite de frog : {ctoto6/toto6}% avec {toto6} ech \n")
    print(f"pourcentage reussite de horse : {ctoto7/toto7}% avec {toto7} ech \n")
    print(f"pourcentage reussite de ship : {ctoto8/toto8}% avec {toto8} ech \n")
    print(f"pourcentage reussite de truck : {ctoto9/toto9}% avec {toto9} ech \n")
start_time= time.time()
predict(0,100)
end_time=time.time()
d_time=end_time-start_time
print(f"Temps écoulés : {d_time:.6} seconde")