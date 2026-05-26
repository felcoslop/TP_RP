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

    from pipeline.thick_roads import run
    run(image_path)

if __name__ == "__main__":
    main()
