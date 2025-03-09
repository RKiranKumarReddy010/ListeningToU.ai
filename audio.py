from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')

text = "Hello Everyone"

generator = pipeline(
    text, voice='af_heart', # <= change voice here
    speed=1, split_pattern=r'\n+'
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    sf.write(f'{i}.wav', audio, 24000) 