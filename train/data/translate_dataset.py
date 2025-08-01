import os
import csv
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_csv(
    input_path: str,
    output_path: str,
    column: str = 'text',
    dest_lang: str = 'es',
    batch_size: int = 64
) -> None:
    # Determinar dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    # Seleccionar modelo según el idioma destino
    if dest_lang == 'es':
        model_name = 'Helsinki-NLP/opus-mt-en-es'
    elif dest_lang == 'en':
        model_name = 'Helsinki-NLP/opus-mt-es-en'
    else:
        print("Solo se soportan traducciones a 'es' o 'en'.")
        raise ValueError("Solo se soportan traducciones a 'es' o 'en'.")

    print(f'Cargando tokenizer y modelo: {model_name} (fp16)...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    if device.type == 'cuda':
        model = model.half()  # usar precisión mixta en GPU
    print('Modelo cargado correctamente en GPU.')

    # Leer archivo de entrada y extraer textos
    print(f'Leyendo CSV de entrada: {input_path}')
    with open(input_path, newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        if column not in reader.fieldnames:
            print(f"Columna '{column}' no encontrada en {input_path}")
            raise ValueError(f"Columna '{column}' no encontrada en {input_path}")
        texts = [row[column] or '' for row in reader]

    total = len(texts)

    # Contar filas ya traducidas
    def count_translated(path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, newline='', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1
    done = count_translated(output_path)
    print(f'Total filas: {total}. Ya traducidas: {done}. Continuando desde {done}.')

    # Inicializar archivo de salida
    fieldnames = ['row_number', column, f'{column}_translated']
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

    # Función de traducción por lotes usando GPU
    def translate_batch(batch_texts: List[str]) -> List[str]:
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Procesar en lotes
    with open(output_path, 'a', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        for start in range(done, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]
            print(f'Traduciendo filas {start} a {end - 1}...')
            translations = translate_batch(batch)
            for idx, (orig, trans) in enumerate(zip(batch, translations), start=start):
                writer.writerow({
                    'row_number': idx,
                    column: orig,
                    f'{column}_translated': trans
                })
            f_out.flush()
            print(f'Procesadas filas {start} a {end - 1}')

    print('Traducción completada.')


def main():
    translate_csv(
        input_path='train/data/english/goemotions_2.csv',
        output_path='train/data/spanish/goemotions_2_translated.csv',
        column='text',
        dest_lang='es',
        batch_size=64
    )

if __name__ == '__main__':
    main()