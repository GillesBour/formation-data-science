from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.sql.types import StringType

def create_pipeline(df, label):
    """
    Crée un pipeline Spark ML pour traiter les colonnes numériques et catégoriques d'un DataFrame.
    Gère automatiquement les cas de classification ou de régression en fonction du type du label.

    :param df: Spark DataFrame d'entrée
    :param label: Nom de la colonne cible
    :return: (pipeline, nom_colonne_label_finale, type_de_problème)
    """
    # Identifier les colonnes numériques et catégoriques
    numeric_cols = [field.name for field in df.schema.fields 
                    if field.dataType.typeName() in ['long', 'double', 'float', 'integer']]  

    categorical_cols = [field.name for field in df.schema.fields 
                        if field.dataType.typeName() == 'string' and field.name != label]

    # Pipeline pour les colonnes numériques
    numeric_pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=numeric_cols, outputCol='numeric_cols_vector'),
        StandardScaler(inputCol='numeric_cols_vector', outputCol='numeric_features')
    ])

    # Pipeline pour les colonnes catégorielles
    index_cols = [f"{col}_idx" for col in categorical_cols]
    ohe_cols = [f"{col}_ohe" for col in categorical_cols]

    categorical_pipeline = Pipeline(stages=[
        StringIndexer(inputCols=categorical_cols, outputCols=index_cols, handleInvalid="keep"),
        OneHotEncoder(inputCols=index_cols, outputCols=ohe_cols, dropLast=True)
    ])

    # Assemblage final
    assembler_input_cols = ['numeric_features'] + ohe_cols
    assembler = VectorAssembler(inputCols=assembler_input_cols, outputCol='features')

    # Détection du type de problème
    is_classification = isinstance(df.schema[label].dataType, StringType)

    stages = [numeric_pipeline, categorical_pipeline, assembler]

    if is_classification:
        stages.append(StringIndexer(inputCol=label, outputCol='label', handleInvalid="keep"))
        label_col = 'label'
        task_type = 'classification'
    else:
        label_col = label  # déjà numérique
        task_type = 'regression'

    final_pipeline = Pipeline(stages=stages)

    return final_pipeline, label_col, task_type
