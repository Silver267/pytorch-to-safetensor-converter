# Pytorch to Safetensor Converter

---



A simple converter which converts pytorch .bin tensor files (Usually listed as "pytorch_model.bin" or "pytorch_model-xxxx-of-xxxx.bin") to safetensor files. Reason? 

~~because it's cool!~~

Because the safetensor format decreases the loading time of large LLM models, currently supported in [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui). It also supports in-place loading, which effectively decreased the required memory to load a LLM.

Note: Most of the code originated from [Convert to Safetensors - a Hugging Face Space by safetensors](https://huggingface.co/spaces/safetensors/convert), and this code cannot deal with files that are not named as "pytorch_model.bin" or "pytorch_model-xxxx-of-xxxx.bin".

### Limitations:

The program requires **A lot** of memory. To be specific, your idle memory should be **at least** twice the size of your largest ".bin" file. Or else, the program will run out of memory and use your swap... that would be **slow!**

This program **will not** re-shard (aka break down) the model, you'll need to do it yourself using some other tools.

### Usage:

After installing python, ``cd`` into the repository and install dependencies first:

```
git clone https://github.com/Silver267/pytorch-to-safetensor-converter.git
cd pytorch-to-safetensor-converter
pip install -r requirements.txt
```

Copy **all content** of your model's folder into this repository, then run:

```
python convert_to_safetensor.py
```
Follow the instruction in the program. Remember to use the **full path** for the model directory (Something like ``E:\models\xxx-fp16`` that contains all the model files). Wait for a while, and you're good to go. The program will automatically copy all other files to your destination folder, enjoy!

### Precision stuff
if your original model is fp32 then don't forget to edit ``"torch_dtype": "float32",`` to ``"torch_dtype": "float16",`` in ``config.json``
#### Note that this operation might (in rare occasions) cause the LLM to output NaN while performing operations since it decreases the precision to fp16.
If you're worried about that, simply edit the line ``loaded = {k: v.contiguous().half() for k, v in loaded.items()}`` in ``convert_to_safetensor.py`` into ``loaded = {k: v.contiguous() for k, v in loaded.items()}`` and you'll have a full precision model.
