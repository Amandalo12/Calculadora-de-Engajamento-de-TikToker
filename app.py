"""
Calculadora de Engajamento de TikToker
--------------------------------------
Aplicação Streamlit que estima o potencial de alcance de um vídeo do TikTok
a partir da categoria e da quantidade de hashtags, usando um modelo TensorFlow
treinado na inicialização com um conjunto de dados SINTÉTICO.

As previsões são estimativas demonstrativas e NÃO representam o desempenho
real da plataforma TikTok.

Executar localmente:  streamlit run app.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Engajamento TikTok",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
SEED = 42
MAX_HASHTAGS = 30
N_AMOSTRAS = 4000

# Alcance mediano (demonstrativo) de cada categoria com número ideal de hashtags
CATEGORIAS = {
    "Dança": 120_000,
    "Humor": 100_000,
    "Beleza": 60_000,
    "Esportes": 50_000,
    "Educação": 35_000,
    "Tecnologia": 30_000,
}
NOMES_CATEGORIAS = list(CATEGORIAS.keys())

LIMITE_FLOPADO = 10_000   # abaixo: Flopado
LIMITE_VIRAL = 100_000    # a partir de: Viral

CLASSES = {
    "Flopado": {
        "cor": "#D64545",
        "fundo": "#FDECEC",
        "icone": "🔴",
        "descricao": "baixo potencial de alcance",
    },
    "Em análise": {
        "cor": "#E0A100",
        "fundo": "#FFF6DB",
        "icone": "🟡",
        "descricao": "potencial de alcance intermediário",
    },
    "Viral": {
        "cor": "#2E9E5B",
        "fundo": "#E7F6EE",
        "icone": "🟢",
        "descricao": "alto potencial de alcance",
    },
}

# ----------------------------------------------------------------------------
# Estilo
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1200px;}
    @media (max-width: 640px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .card-valor {font-size: 1.9rem;}
    }
    .titulo {font-size: 2rem; font-weight: 700; color: #1F2A44; margin-bottom: 0;}
    .subtitulo {color: #5B6478; margin-top: 0.2rem; margin-bottom: 1rem;}
    .card {
        border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.08);
    }
    .card-rotulo {font-size: 0.85rem; letter-spacing: .06em; text-transform: uppercase;
                  color: #5B6478; margin: 0;}
    .card-valor {font-size: 2.4rem; font-weight: 800; margin: 0.1rem 0 0 0; line-height: 1.1;}
    .card-faixa {color: #5B6478; margin-top: 0.3rem; font-size: 0.95rem;}
    .selo {display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
           font-weight: 700; color: #fff; font-size: 1rem; margin-bottom: 0.6rem;}
    .medidor {position: relative; height: 14px; border-radius: 999px; margin: 1.6rem 0 0.4rem 0;
              background: linear-gradient(90deg,#D64545 0%,#D64545 33%,#E0A100 33%,#E0A100 67%,#2E9E5B 67%,#2E9E5B 100%);}
    .marcador {position: absolute; top: -7px; width: 4px; height: 28px; background: #1F2A44;
               border-radius: 2px;}
    .escala {display: flex; justify-content: space-between; font-size: 0.78rem; color: #5B6478;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Dados sintéticos
# ----------------------------------------------------------------------------
def fator_hashtags(hashtags):
    """Efeito demonstrativo das hashtags: pico perto de 6, queda com excesso."""
    hashtags = np.asarray(hashtags, dtype="float64")
    return 0.25 + np.exp(-((hashtags - 6.0) ** 2) / (2 * 3.5**2))


def gerar_dados_sinteticos(n=N_AMOSTRAS, seed=SEED):
    """Gera uma base fictícia (categoria, hashtags, alcance) para demonstração."""
    rng = np.random.default_rng(seed)
    idx_cat = rng.integers(0, len(NOMES_CATEGORIAS), size=n)
    hashtags = rng.integers(0, MAX_HASHTAGS + 1, size=n)
    base = np.array(list(CATEGORIAS.values()), dtype="float64")[idx_cat]
    ruido = rng.lognormal(mean=0.0, sigma=0.25, size=n)
    alcance = base * fator_hashtags(hashtags) * ruido
    return pd.DataFrame(
        {"categoria_idx": idx_cat, "hashtags": hashtags, "alcance": alcance}
    )


# ----------------------------------------------------------------------------
# Modelo TensorFlow
# ----------------------------------------------------------------------------
def preprocessar(indices_categoria, hashtags):
    """Transforma as entradas: one-hot da categoria (tf.one_hot) e hashtags float."""
    cat = tf.one_hot(
        np.asarray(indices_categoria, dtype="int32"),
        depth=len(NOMES_CATEGORIAS),
        dtype=tf.float32,
    )
    hs = tf.constant(
        np.asarray(hashtags, dtype="float32").reshape(-1, 1), dtype=tf.float32
    )
    return {"categoria": cat, "hashtags": hs}


@st.cache_resource(show_spinner="Treinando o modelo demonstrativo com dados sintéticos...")
def treinar_modelo():
    """Treina o modelo uma única vez por processo e o mantém em cache."""
    tf.keras.utils.set_random_seed(SEED)

    df = gerar_dados_sinteticos()
    y = np.log10(df["alcance"].to_numpy(dtype="float32"))
    y_media, y_desvio = float(y.mean()), float(y.std())
    y_norm = ((y - y_media) / y_desvio).astype("float32")

    entradas = preprocessar(df["categoria_idx"].to_numpy(), df["hashtags"].to_numpy())

    # Separação treino (80%) / validação (20%)
    n = len(df)
    corte = int(n * 0.8)
    treino_x = {k: v[:corte] for k, v in entradas.items()}
    valid_x = {k: v[corte:] for k, v in entradas.items()}
    treino_y, valid_y = y_norm[:corte], y_norm[corte:]

    # Arquitetura: categoria (one-hot) + hashtags (normalizadas) -> MLP
    in_cat = tf.keras.Input(shape=(len(NOMES_CATEGORIAS),), name="categoria")
    in_hash = tf.keras.Input(shape=(1,), name="hashtags")
    normalizador = tf.keras.layers.Normalization(axis=-1)
    normalizador.adapt(treino_x["hashtags"].numpy())
    h = normalizador(in_hash)
    x = tf.keras.layers.Concatenate()([in_cat, h])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    saida = tf.keras.layers.Dense(1)(x)
    modelo = tf.keras.Model(inputs=[in_cat, in_hash], outputs=saida)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mae"],
    )
    modelo.fit(
        treino_x,
        treino_y,
        validation_data=(valid_x, valid_y),
        epochs=80,
        batch_size=64,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
        ],
    )

    _, mae_norm = modelo.evaluate(valid_x, valid_y, verbose=0)
    mae_log10 = float(mae_norm) * y_desvio
    return {
        "modelo": modelo,
        "y_media": y_media,
        "y_desvio": y_desvio,
        "fator_erro": float(10**mae_log10),
        "n_amostras": n,
    }


def prever_alcance(info, indices_categoria, hashtags):
    """Retorna o alcance estimado (em visualizações) para um ou vários exemplos."""
    entradas = preprocessar(indices_categoria, hashtags)
    saida = info["modelo"](entradas, training=False).numpy().ravel()
    return 10 ** (saida * info["y_desvio"] + info["y_media"])


# ----------------------------------------------------------------------------
# Apoio à interface
# ----------------------------------------------------------------------------
def formatar_numero(valor):
    return f"{valor:,.0f}".replace(",", ".")


def classificar(alcance):
    if alcance < LIMITE_FLOPADO:
        return "Flopado"
    if alcance < LIMITE_VIRAL:
        return "Em análise"
    return "Viral"


def posicao_medidor(alcance):
    """Posição (0-100) em escala logarítmica de 1 mil a 1 milhão de visualizações."""
    pos = (np.log10(max(alcance, 1.0)) - 3.0) / 3.0
    return float(np.clip(pos, 0.02, 0.98) * 100)


def explicar(classe, categoria, hashtags):
    texto = (
        f"Para um vídeo da categoria **{categoria}** com **{hashtags}** "
        f"hashtag(s), o modelo demonstrativo indica **{CLASSES[classe]['descricao']}**. "
    )
    if hashtags < 4:
        texto += (
            "Poucas hashtags tendem a reduzir a descoberta do conteúdo nesta simulação."
        )
    elif hashtags <= 9:
        texto += (
            "A quantidade de hashtags está na faixa que, nesta simulação, "
            "se associa aos melhores resultados."
        )
    elif hashtags <= 15:
        texto += (
            "A quantidade de hashtags já é elevada e o ganho adicional tende a ser pequeno."
        )
    else:
        texto += (
            "O excesso de hashtags tende a reduzir o alcance estimado nesta simulação."
        )
    return texto


def validar_entradas(hashtags):
    """Retorna mensagem de erro amigável ou None se estiver tudo válido."""
    if hashtags is None:
        return "Informe a quantidade de hashtags para calcular a previsão."
    try:
        valor = float(hashtags)
    except (TypeError, ValueError):
        return "A quantidade de hashtags deve ser um número."
    if not np.isfinite(valor):
        return "A quantidade de hashtags deve ser um número válido."
    if not valor.is_integer():
        return "A quantidade de hashtags deve ser um número inteiro."
    if valor < 0:
        return "A quantidade de hashtags não pode ser negativa."
    if valor > MAX_HASHTAGS:
        return f"A quantidade máxima de hashtags aceita é {MAX_HASHTAGS}."
    return None


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.markdown(
    '<p class="titulo">📈 Calculadora de Engajamento de TikToker</p>'
    '<p class="subtitulo">Estimativa de potencial de alcance para apoiar a análise '
    "de campanhas e a seleção de influenciadores.</p>",
    unsafe_allow_html=True,
)
st.info(
    "**Ferramenta demonstrativa.** As previsões abaixo são estimativas geradas por um "
    "modelo treinado com dados sintéticos e **não representam o desempenho real do TikTok**."
)

info_modelo = treinar_modelo()

coluna_form, coluna_resultado = st.columns([1, 1.4], gap="large")

with coluna_form:
    st.subheader("Dados do vídeo")
    with st.form("form_previsao"):
        categoria = st.selectbox(
            "Categoria do vídeo",
            NOMES_CATEGORIAS,
            help="Tema principal do conteúdo publicado.",
        )
        hashtags = st.number_input(
            "Quantidade de hashtags",
            min_value=0,
            max_value=MAX_HASHTAGS,
            value=5,
            step=1,
            help=f"Informe um número inteiro entre 0 e {MAX_HASHTAGS}.",
        )
        enviado = st.form_submit_button(
            "Calcular previsão", type="primary", use_container_width=True
        )

    if enviado:
        erro = validar_entradas(hashtags)
        if erro:
            st.session_state.pop("resultado", None)
            st.error(erro)
        else:
            st.session_state["resultado"] = {
                "categoria": categoria,
                "hashtags": int(hashtags),
            }

with coluna_resultado:
    st.subheader("Resultado da previsão")
    resultado = st.session_state.get("resultado")

    if not resultado:
        st.write(
            "Selecione a categoria, informe a quantidade de hashtags e clique em "
            "**Calcular previsão**."
        )
    else:
        cat = resultado["categoria"]
        n_hash = resultado["hashtags"]
        idx = NOMES_CATEGORIAS.index(cat)

        alcance = float(prever_alcance(info_modelo, [idx], [n_hash])[0])
        classe = classificar(alcance)
        estilo = CLASSES[classe]
        fator = info_modelo["fator_erro"]
        faixa_min, faixa_max = alcance / fator, alcance * fator

        html_card = (
            f'<div class="card" style="background:{estilo["fundo"]}; border-left: 6px solid {estilo["cor"]};">'
            f'<span class="selo" style="background:{estilo["cor"]};">{estilo["icone"]} {classe}</span>'
            f'<p class="card-rotulo">Alcance estimado (demonstrativo)</p>'
            f'<p class="card-valor" style="color:{estilo["cor"]};">{formatar_numero(alcance)}</p>'
            f'<p class="card-faixa">Faixa provável: {formatar_numero(faixa_min)} a {formatar_numero(faixa_max)} visualizações</p>'
            f'<div class="medidor"><div class="marcador" style="left:{posicao_medidor(alcance):.1f}%;"></div></div>'
            f'<div class="escala"><span>1 mil</span><span>{formatar_numero(LIMITE_FLOPADO)}</span>'
            f'<span>{formatar_numero(LIMITE_VIRAL)}</span><span>1 mi+</span></div>'
            f'</div>'
        )
        st.markdown(html_card, unsafe_allow_html=True)

        st.markdown(explicar(classe, cat, n_hash))

        aba1, aba2 = st.tabs(["Efeito das hashtags", "Comparar categorias"])

        with aba1:
            qtd = np.arange(0, MAX_HASHTAGS + 1)
            curva = prever_alcance(info_modelo, np.full(qtd.shape, idx), qtd)
            df_curva = pd.DataFrame(
                {"Alcance estimado": curva}, index=pd.Index(qtd, name="Hashtags")
            )
            st.line_chart(df_curva)
            st.caption(f"Curva estimada para a categoria {cat}.")

        with aba2:
            idxs = np.arange(len(NOMES_CATEGORIAS))
            valores = prever_alcance(info_modelo, idxs, np.full(idxs.shape, n_hash))
            df_cat = pd.DataFrame(
                {"Alcance estimado": valores},
                index=pd.Index(NOMES_CATEGORIAS, name="Categoria"),
            )
            st.bar_chart(df_cat)
            st.caption(f"Comparação entre categorias com {n_hash} hashtag(s).")

st.divider()
with st.expander("Sobre o modelo e as limitações"):
    st.markdown(
        f"""
- **Classificação:** *Flopado* (< {formatar_numero(LIMITE_FLOPADO)} visualizações estimadas),
  *Em análise* (entre {formatar_numero(LIMITE_FLOPADO)} e {formatar_numero(LIMITE_VIRAL)}) e
  *Viral* (≥ {formatar_numero(LIMITE_VIRAL)}).
- **Modelo:** rede neural em TensorFlow/Keras (categoria em *one-hot*, hashtags normalizadas),
  treinada na inicialização com {formatar_numero(info_modelo['n_amostras'])} registros sintéticos.
- **Erro típico na validação:** o valor real simulado costuma ficar entre
  {1 / info_modelo['fator_erro']:.2f}x e {info_modelo['fator_erro']:.2f}x da previsão.
- **Importante:** os dados são fictícios. Para previsões confiáveis em produção, é necessário
  treinar e validar o modelo com dados reais de campanhas e resultados de vídeos.
        """
    )
