from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()

long_text = """
Ninguém sabe ao certo quem é o herói ou o vilão desse fascinante universo criado por Patrick Rothfuss. Na realidade, essas duas figuras se concentram em Kote, um homem enigmático que se esconde sob a identidade de proprietário da hospedaria Marco do Percurso. Da infância numa trupe de artistas itinerantes, passando pelos anos vividos numa cidade hostil e pelo esforço para ingressar na escola de magia, O nome do vento acompanha a trajetória de Kote e as duas forças que movem sua vida: o desejo de aprender o mistério por trás da arte de nomear as coisas e a necessidade de reunir informações sobre o Chandriano – os lendários demônios que assassinaram sua família no passado. Quando esses seres do mal reaparecem na cidade, um cronista suspeita de que o misterioso Kote seja o personagem principal de diversas histórias que rondam a região e decide aproximar-se dele para descobrir a verdade. Pouco a pouco, a história de Kote vai sendo revelada, assim como sua multifacetada personalidade – notório mago, esmerado ladrão, amante viril, herói salvador, músico magistral, assassino infame. Nesta provocante narrativa, o leitor é transportado para um mundo fantástico, repleto de mitos e seres fabulosos, heróis e vilões, ladrões e trovadores, amor e ódio, paixão e vingança. 
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=70)

parts = splitter.create_documents([long_text])

llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

chain_summarize = load_summarize_chain(llm_en, chain_type="map_reduce", verbose=False)

result = chain_summarize.invoke({"input_documents": parts})

print(result["output_text"])