import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Uso: python run_thick.py <caminho_da_imagem>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Erro: Arquivo '{image_path}' nao encontrado.")
        sys.exit(1)

    # Busca dinâmica do arquivo de pesos
    checkpoint = "weights/cityscale_vitb_512_e10.ckpt"
    if not os.path.exists(checkpoint):
        fallback = "../weights/cityscale_vitb_512_e10.ckpt"
        if os.path.exists(fallback):
            checkpoint = fallback
        else:
            print("Erro: Checkpoint 'cityscale_vitb_512_e10.ckpt' nao encontrado.")
            print("Por favor, crie a pasta 'weights/' e coloque o arquivo nela.")
            print("Consulte 'documentacao_pesos.md' para os links de download e instrucoes.")
            sys.exit(1)

    print(f"Usando checkpoint: {checkpoint}")
    from pipeline.thick_roads import run
    run(image_path, checkpoint=checkpoint)

if __name__ == "__main__":
    main()
