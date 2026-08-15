import Config_Bot

# ===========================================================
# 🧑‍💻 ÁREA DO ALUNO
# ============================================================

def meu_agente(percepcao: Config_Bot.Perception) -> str:
    # TODO 1: se estiver sujo, retorne "Suck".
    if percepcao[1] == "Dirty":
        return "Suck"
    # TODO 2: se estiver limpo, escolha "Right" ou "Left".
    if percepcao[0] == "A":
            return "Right"
    else: 
         return "Left"

display = print

test_perceptions = [
    ("A", "Dirty"), ("A", "Clean"),
    ("B", "Dirty"), ("B", "Clean")
]

for perception in test_perceptions:
    print(perception, "→", meu_agente(perception))

demo = Config_Bot.run_episode(
    meu_agente,
    start_position="A",
    status_map={"A": "Dirty", "B": "Dirty"}
)

print("Ações:", demo.actions)
print("Estado final:", demo.status)
print("Sucesso:", demo.is_clean())
print("Passos:", demo.steps)

rows = []

for episode, (start, status_map) in enumerate(Config_Bot.all_states(), start=1):
    world = Config_Bot.run_episode(meu_agente, start, status_map)

    rows.append({
        "episódio": episode,
        "início": start,
        "mapa": status_map,
        "sucesso": world.is_clean(),
        "passos": world.steps,
        "ações": world.actions,
    })

results = Config_Bot.pd.DataFrame(rows)
display(results)

print(f"Resultado: {results['sucesso'].sum()} / {len(results)} estados resolvidos.")