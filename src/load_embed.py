import gensim
import zipfile
import json

def load_embedding(modelfile=None, input_type = "tokenized"):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile is None:
        if input_type == "lemmatized":
            modelfile = "223.zip"
            #modelfile = "/cluster/shared/nlpl/data/vectors/latest/223.zip"
        elif input_type == "tokenized":
            modelfile = "222.zip"
            #modelfile = "/cluster/shared/nlpl/data/vectors/latest/222.zip"

    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
            modelfile.endswith(".txt.gz")
            or modelfile.endswith(".txt")
            or modelfile.endswith(".vec.gz")
            or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open("meta.json")
            metadata = json.loads(metafile.read())
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace",  limit=200000
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (optional):
    emb_model.init_sims(
        replace=True
    )
    return emb_model