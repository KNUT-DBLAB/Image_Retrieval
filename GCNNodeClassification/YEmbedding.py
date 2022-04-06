import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from visual_genome import api as vg
import urllib.request
from PIL import Image
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.cluster import KMeansClusterer
from scipy.spatial import distance_matrix, distance


''' 1000개의 이미지 별로 region_graph 의 phrase 값을 embedding 하고, 각 이미지를 15개의 클러스터로 분류함
    각 이미지별 클러스터와 centroid 값을 추출하면 좋을 듯(centroid를 학습 시 feature map 에 반영 할 수 있나?(처음에 랜덤값 주고 centroid에 영향 받게끔 해서?))'''



def visualize_regions(image, regions):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    img = Image.open(urllib.request.urlopen(image.url))  # url from API
    plt.imshow(img)
    ax = plt.gca()
    for region in regions:  # visualize regions
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region.x, region.y, region.phrase, style='italic', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


def retrieve_image(image_id):
    # find the image from api
    image = vg.get_image_data(id=image_id)  # VG api works here
    if image:
        print("Image data:", image)
        reg = vg.get_region_descriptions_of_image(id=image_id)
        # show images
        visualize_regions(image, reg[:8])  # call with fewer regions for better visualization


def get_embeddings(model, region_sentences):
    sentence_embeddings = model.encode(region_sentences)
    return sentence_embeddings


def clustering_question(images_regions, key, NUM_CLUSTERS):
    sentences = images_regions['region_sentences']
    X = np.array(images_regions[key].tolist())
    data = images_regions[['image_id', 'region_sentences', key]].copy()

    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(data[key], assign_clusters=True)

    data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])

    return data, assigned_clusters


def nltk_inertia(feature_matrix, centroid):
    sum_ = []
    for i in range(feature_matrix.shape[0]):
        sum_.append(np.sum((feature_matrix[i] - centroid[i]) ** 2))

    return sum(sum_)


def number_of_clusters(image_regions, key):
    sse = []
    list_k = list(range(10, 500, 50))

    for k in list_k:
        data, assigned_clusters = clustering_question(image_regions, key, k)
        sse.append(nltk_inertia(data[key].to_numpy(), data.centroid.to_numpy()))

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.title('Elbow method for ' + key)
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters k')
    plt.ylabel('Sum of squared distance')
    plt.show()


"""
for key in key_list:
    number_of_clusters(image_regions, key)
"""


def make_clusters(key, images_regions, n_clusters):
    data, assigned_clusters = clustering_question(images_regions, key, NUM_CLUSTERS=n_clusters)
    # Compute centroid distance to the data
    data['distance_from_centroid'] = data.apply(distance_from_centroid, axis=1)
    return data


def distance_from_centroid(row):
    # type of emb and centroid is different, hence using tolist below
    return distance_matrix([row['embeddings_1']], [row['centroid'].tolist()])[0][0]


def find_distances(images_regions, input_id, key, input_embedding=np.zeros(5)):
    distances = []
    image_ids = images_regions.Image_id
    if input_id > 0:
        reference_embedding = images_regions.loc[images_regions.Image_id == input_id][key]
        reference_embedding = reference_embedding.values[0].reshape(-1, 1)
        corpus_embeddings = images_regions.loc[images_regions.Image_id != input_id][key]
    else:
        reference_embedding = input_embedding
        corpus_embeddings = images_regions[key]

    for j in range(len(corpus_embeddings)):  # rows of def_embeddings matrix
        defin = j
        if image_ids[j] != input_id:  # avoid calculating distance with itself
            corpus = corpus_embeddings[j].reshape(-1, 1)
            # euclidean distance between multidimensional vectors
            dist = distance.euclidean(reference_embedding, corpus)
            distances.append([image_ids[j], dist])

            # store in df
    col_names = ['image_id', 'distances']
    distances_df = pd.DataFrame(distances, columns=col_names)
    distances_df = distances_df.sort_values(by='distances', ascending=True)
    distances_df.to_csv('distances.csv', index=False)
    return distances_df


# given image id to retrieve its k most similar images

def retrieve_images(key, images_regions, input_id=-1, input_embedding=np.zeros(5)):
    print('Images retrieved using method:', key)
    top_k = 10

    if input_id > 0:
        retrieve_image(input_id)
        distances_df = find_distances(images_regions, input_id, key)
    else:
        distances_df = find_distances(images_regions, input_id, key, input_embedding)
    top_images = distances_df.head(top_k)

    print("Top", top_k, "most similar images to image", input_id, "in Visual Genome:")
    for index, row in top_images.iterrows():
        im_id = int(row.image_id)

        # find similar images from api and show
        retrieve_image(im_id)
    return top_images


# xlxspath : './data/image_regions.xlsx'
def YEmbedding(xlxspath):
    # pre-trained models
    model_1 = SentenceTransformer('bert-base-nli-mean-tokens')
    models = [model_1]

    ## Read images and their descriptions
    image_regions = pd.read_excel(xlxspath)

    for i in range(len(image_regions['region_sentences'])):
        image_regions["region_sentences"][i] = image_regions["region_sentences"][i].split(',')

    regions = image_regions["region_sentences"].tolist()

    key_list = ["embeddings_1"]
    embeddings = dict.fromkeys(key_list)
    for model, key in zip(models, key_list):
        values = []
        for corpus in regions:
            # returns a list of lists, one 768-length embedding per region sentence.
            # Number of rows corresponds to number of sentences
            if corpus:
                emb = get_embeddings(model, corpus)
                # get mean value columnwise, so that sentence embeddings are averaged per region for each image
                emb = np.mean(np.array(emb), axis=0)
                # for each model, a 768-length embedding is stored
                values.append(emb)
            else:
                values.append(0)
                print(0)
        embeddings[key] = values

    for key in key_list:
        image_regions[key] = pd.Series(embeddings[key], index=image_regions.index)
        # number of sentences x 768
    image_regions.head()

    n_clusters = 15
    embeddings_method = "embeddings_1"
    pd.set_option('display.max_columns', None)

    #print(len)
    #print(image_regions)
    #print("img_regions : ", image_regions.head(5))
    embedding_clusters = make_clusters(embeddings_method, image_regions, n_clusters)

    pd.set_option('display.max_columns', None)
    #print("embedding_cluster : ", embedding_clusters.head(5))
    #print("embedding_cluster : ", embedding_clusters[['cluster']].head(5))

    return embedding_clusters
