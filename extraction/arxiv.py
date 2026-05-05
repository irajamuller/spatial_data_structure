import json
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from keybert import KeyBERT


def extrair_dados_arxiv_json(base_url, arquivo_saida='artigos_arxiv_spatial.json'):
    """
    Extrai dados de artigos do ArXiv paginando de 200 em 200 e salva em formato JSON.

    Args:
        base_url (str): A URL base da busca avançada do ArXiv (sem o parâmetro 'start').
        arquivo_saida (str): O nome do arquivo JSON para salvar os resultados.
    """
    
    kw_model = KeyBERT('all-MiniLM-L6-v2')

    # Lista para armazenar todos os artigos
    todos_os_artigos = []
    
    # Inicializa o offset da paginação
    start = 0
    
    # Parâmetros de cabeçalho para simular um navegador e evitar bloqueios
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("Iniciando a extração de dados do ArXiv (Saída JSON)...")
    
    while True:
        # Constrói a URL para a página atual
        url_pagina = f"{base_url}&start={start}"
        print(f"\nAcessando página com offset 'start': {start}")

        try:
            # 1. Faz a requisição HTTP
            response = requests.get(url_pagina, headers=headers, timeout=15)
            response.raise_for_status() # Lança um erro para status de erro (4xx ou 5xx)
            
            # 2. Analisa o HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 3. Encontra todos os blocos de resultados (artigos)
            artigos_na_pagina = soup.find_all('li', class_='arxiv-result')
            
            # Verifica se há artigos na página. Se não houver, a extração terminou.
            if not artigos_na_pagina:
                print("Não foram encontrados mais artigos. Extração concluída.")
                break
                
            print(f"Encontrados {len(artigos_na_pagina)} artigos nesta página.")

            # 4. Processa cada artigo
            for artigo in artigos_na_pagina:
                dados_artigo = {}
                
                # Título (incluído para melhor referência)
                title_tag = artigo.find('p', class_='title')
                dados_artigo['titulo'] = title_tag.text.strip() if title_tag else 'N/A'
                
                # --- Autores ---
                authors_div = artigo.find('p', class_='authors')
                if authors_div:
                    # Remove o prefixo "Authors:" e extrai o texto
                    authors_text = authors_div.text.replace('Authors:', '').strip()
                    # Limpa a string de autores, removendo excesso de espaços e quebras de linha
                    dados_artigo['autores'] = [x.strip() for x in re.sub(r'\s+', ' ', authors_text).strip().split(',')]
                else:
                    dados_artigo['autores'] = 'N/A'

                # --- Abstract ---
                abstract_div = artigo.find('span', class_='abstract-full')
                if abstract_div:
                    abstract_text = abstract_div.text.strip()
                    if abstract_text.startswith('Abstract:'):
                         abstract_text = abstract_text[len('Abstract:'):].strip()
                    dados_artigo['abstract'] = abstract_text
                else:
                    dados_artigo['abstract'] = 'N/A'

                dados_artigo['keywords'] = [kw for kw, score in kw_model.extract_keywords(dados_artigo['abstract'], keyphrase_ngram_range=(1, 3), top_n=3)]

                # Categorias
                dados_artigo["categorias"] = []
                cat_spans = artigo.find_all("span", class_="tag is-small is-link tooltip is-tooltip-top")
                if cat_spans:
                    categories = [span.text.strip() for span in cat_spans] if cat_spans else []
                    dados_artigo["categorias"] = categories                

                cat_spans = artigo.find_all("span", class_="tag is-small is-grey tooltip is-tooltip-top")
                if cat_spans:
                    categories = [span.text.strip() for span in cat_spans] if cat_spans else []
                    dados_artigo["categorias"] = dados_artigo["categorias"] + categories                

                # --- Ano (Submissão) ---
                metadata_p = artigo.find("p", class_="is-size-7")
                dados_artigo['ano'] = 'N/A'
                if metadata_p:
                    # Procura pela data de submissão (e.g., [Submitted on 1 Jan 2020])
                    #submitted_match = re.search(r'Submitted\s+on\s+\d{1,2}\s+\w{3}\s+(\d{4})', metadata_p.text)
                    #submitted_match = re.search(r'Submitted\s+.*?,\s*(\d{4})', metadata_p.text)
                    match = re.search(r"\b(20\d{2})\b", metadata_p.text.split(";")[0])
                    year = int(match.group(1)) if match else None
                    if year:
                        dados_artigo['ano'] = year
                    else:
                        dados_artigo['ano'] = 'N/A'

                metadata_p = artigo.find_all("p", class_="comments is-size-7")
                dados_artigo['journal_revista'] = 'N/A'
                if metadata_p:
                    for p in metadata_p:
                        span = p.find("span")
                        if span and "Journal ref" in span.get_text():
                            journal_ref = p.get_text().replace("Journal ref:", "").strip().strip('"').strip()
                            dados_artigo['journal_revista'] = journal_ref
                
                # Adiciona o artigo à lista principal
                todos_os_artigos.append(dados_artigo)

            # 5. Prepara para a próxima página
            start += 200
            
            # Pausa para ser gentil com o servidor
            time.sleep(3) 
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição: {e}")
            break
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")
            break

    # 6. Salva os dados em um arquivo JSON
    if todos_os_artigos:
        try:
            with open(arquivo_saida, 'w', encoding='utf-8') as f:
                # Usa 'indent=4' para formatar o JSON de forma legível
                json.dump(todos_os_artigos, f, ensure_ascii=False, indent=4)
            print(f"\n--- EXTRAÇÃO CONCLUÍDA ---")
            print(f"Total de artigos extraídos: {len(todos_os_artigos)}")
            print(f"Dados salvos em '{arquivo_saida}'")
        except IOError as e:
            print(f"Erro ao salvar o arquivo JSON: {e}")
    else:
        print("\nNão foi possível extrair nenhum artigo.")

# --- Execução do Algoritmo ---
BASE_URL_DO_USUARIO = "https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=%22spatial+data+structure%22&terms-0-field=abstract&terms-1-operator=OR&terms-1-term=%22geospatial+data+structure%22&terms-1-field=abstract&terms-2-operator=OR&terms-2-term=%22quadtree%22&terms-2-field=abstract&terms-3-operator=OR&terms-3-term=%22r-tree%22&terms-3-field=abstract&terms-4-operator=OR&terms-4-term=%22m-tree%22&terms-4-field=abstract&terms-5-operator=OR&terms-5-term=%22kd-tree%22&terms-5-field=title&terms-6-operator=AND&terms-6-term=%22spatial+index%22&terms-6-field=title&classification-computer_science=y&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=200&order=-submitted_date"


extrair_dados_arxiv_json(BASE_URL_DO_USUARIO)
