# Pytorch to Safetensor Converter

---



A simple converter which converts pytorch .bin tensor files (Usually listed as "pytorch_model.bin" or "pytorch_model-xxxx-of-xxxx.bin") to safetensor files. Why? 

~~because it's cool!~~

Because it decreases the loading time of large LLM models, currently supported in [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui). 

Note: Most of the code originated from [Convert to Safetensors - a Hugging Face Space by safetensors](https://huggingface.co/spaces/safetensors/convert), and this code cannot deal with files that are not named as "pytorch_model.bin" or "pytorch_model-xxxx-of-xxxx.bin".

### Limitations:

The program requires **A lot** of memory. To be specific, your idle memory should be **at least** the size of your largest ".bin" file. Or else, the program will run out of memory and use your swap... that would be **slow!**

This program **will not** re-shard (aka break down) the model, you'll need to do it yourself using some other tools.

### Usage:

After installing python, cd into the repository and install dependencies first:

```
git clone https://github.com/Silver267/pytorch-to-safetensor-converter.git
cd pytorch-to-safetensor-converter
pip install -r requirements.txt
```

Copy **all content** of your model's folder into this repository, then run:

```
python convert_to_safetensor.py
```

After inputting your target folder and waiting for some time for the conversion to be over, you should see a bunch (or one if you only have one pytorch_model.bin) of .safetensor files, these are the converted files. Additionally, you will also see a file that looks like: "model.safetensors.index.json", which is the mapping of the model, also important.

After the conversion is completed, copy:

```
config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

Into your target folder, then copy your target folder to the place you want it to be (models folder if you're using oobabooga's webui), and you're good to go!


### (edit: the operation below might cause the LLM to output NaN due to precision stuff... so use with caution!)

if your original model is fp32 then don't forget to edit the
```
"torch_dtype": "float32",
```
to
```
"torch_dtype": "float16",
```
in
```
config.json
```
