#!/bin/bash

# Diretorios locais do Greengrass
GG_ROOT="/greengrass/v2"
COMPONENT_NAME="com.manometro.leitor"
COMPONENT_VERSION="1.1.0"

RECIPES_DIR="/home/luiz/greengrass-build/recipes"
ARTIFACTS_DIR="/home/luiz/greengrass-build/artifacts/$COMPONENT_NAME/$COMPONENT_VERSION"

echo "1. Criando estrutura de pastas do componente local..."
mkdir -p $RECIPES_DIR
mkdir -p $ARTIFACTS_DIR

echo "2. Copiando arquivos..."
# A receita 
cp /home/luiz/ProyectoManometro/com.manometro.leitor-1.1.0.yaml $RECIPES_DIR/

# O codigo Python
cp /home/luiz/ProyectoManometro/gauge_reader.py $ARTIFACTS_DIR/

# A pasta web (templates e estaticos)
cp -r /home/luiz/ProyectoManometro/web $ARTIFACTS_DIR/

# Copiando o modelo compilado (segmentacion.engine) para a pasta de artefatos
cp "/home/luiz/ProyectoManometro/segmentacion.engine" $ARTIFACTS_DIR/

echo "3. Fazendo Deploy do componente usando Greengrass CLI..."
sudo /greengrass/v2/bin/greengrass-cli deployment create \
  --recipeDir $RECIPES_DIR \
  --artifactDir /home/luiz/greengrass-build/artifacts \
  --merge "$COMPONENT_NAME=$COMPONENT_VERSION"

echo "========================================================="
echo "Deploy finalizado!"
echo "Verifique os logs de execucao com o comando:"
echo "sudo tail -f /greengrass/v2/logs/com.manometro.leitor.log"
echo "========================================================="
