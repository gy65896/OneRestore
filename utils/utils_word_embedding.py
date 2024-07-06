import torch
import numpy as np
import fasttext.util
from gensim import models

def load_word_embeddings(emb_file, vocab):
    embeds = {}
    for line in open(emb_file, 'rb'):
        line = line.decode().strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        embeds[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {
        'Faux.Fur':'fake_fur', 'Faux.Leather':'fake_leather', 'Full.grain.leather':'thick_leather', 
        'Hair.Calf':'hair_leather', 'Patent.Leather':'shiny_leather', 'Nubuck':'grainy_leather', 
        'Boots.Ankle':'ankle_boots', 'Boots.Knee.High':'knee_high_boots', 'Boots.Mid-Calf':'midcalf_boots', 
        'Shoes.Boat.Shoes':'boat_shoes', 'Shoes.Clogs.and.Mules':'clogs_shoes', 'Shoes.Flats':'flats_shoes',
        'Shoes.Heels':'heels', 'Shoes.Loafers':'loafers', 'Shoes.Oxfords':'oxford_shoes',
        'Shoes.Sneakers.and.Athletic.Shoes':'sneakers'}
    custom_map_vaw = {
        'selfie': 'photo'
    }

    E = []
    for k in vocab:
        if k in custom_map:
            print(f'Change {k} to {custom_map[k]}')
            k = custom_map[k]
        k = k.lower()
        if '_' in k:
            toks = k.split('_')
            emb_tmp = torch.zeros(300).float()
            for tok in toks:
                if tok in custom_map_vaw:
                    tok = custom_map_vaw[tok]
                emb_tmp += embeds[tok]
            emb_tmp /= len(toks)
            E.append(emb_tmp)
        else:
            E.append(embeds[k])

    embeds = torch.stack(E)
    print ('Loaded embeddings from file %s' % emb_file, embeds.size())

    return embeds

def load_fasttext_embeddings(emb_file,vocab):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }
    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)

    
    ft = fasttext.load_model(emb_file) #DATA_FOLDER+'/fast/cc.en.300.bin')
    embeds = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)

    embeds = torch.Tensor(np.stack(embeds))
    print('Fasttext Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds

def load_word2vec_embeddings(emb_file,vocab):
    # vocab = [v.lower() for v in vocab]

    
    model = models.KeyedVectors.load_word2vec_format(emb_file,binary=True)
        #DATA_FOLDER+'/w2v/GoogleNews-vectors-negative300.bin', binary=True)

    custom_map = {
        'Faux.Fur': 'fake_fur',
        'Faux.Leather': 'fake_leather',
        'Full.grain.leather': 'thick_leather',
        'Hair.Calf': 'hair_leather',
        'Patent.Leather': 'shiny_leather',
        'Boots.Ankle': 'ankle_boots',
        'Boots.Knee.High': 'knee_high_boots',
        'Boots.Mid-Calf': 'midcalf_boots',
        'Shoes.Boat.Shoes': 'boat_shoes',
        'Shoes.Clogs.and.Mules': 'clogs_shoes',
        'Shoes.Flats': 'flats_shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford_shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k and k not in model:
            ks = k.split('_')
            emb = np.stack([model[it] for it in ks]).mean(axis=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    print('Word2Vec Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds



def initialize_wordembedding_matrix(name, vocab):
    """
    Args:
    - name: hyphen separated word embedding names: 'glove-word2vec-conceptnet'.
    - vocab: list of attributes/objects.
    """
    wordembs = name.split('+')
    result = None

    for wordemb in wordembs:
        if wordemb == 'glove':
            wordemb_ = load_word_embeddings(f'./utils/glove.6B.300d.txt', vocab)
        if result is None:
            result = wordemb_
        else:
            result = torch.cat((result, wordemb_), dim=1)
    dim = 300 * len(wordembs)
    return result, dim
