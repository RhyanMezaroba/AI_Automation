import Config_Bot_5Salas

# ============================================================
# 🧑‍💻 ÁREA DO ALUNO — 5 SALAS
# ============================================================

def meu_agente_5(percepcao: Perception5) -> str:
    # TODO 1: se estiver sujo, limpe.
    # TODO 2: quando estiver limpo, escolha uma política de movimento.
    return "NoOp"

PUBLIC_SCENARIOS = [
    ("P01", "C", {"A":"Clean","B":"Dirty","C":"Clean","D":"Clean","E":"Clean"}),
    ("P02", "C", {"A":"Clean","B":"Clean","C":"Clean","D":"Dirty","E":"Clean"}),
    ("P03", "C", {"A":"Dirty","B":"Clean","C":"Clean","D":"Clean","E":"Clean"}),
    ("P04", "C", {"A":"Clean","B":"Clean","C":"Clean","D":"Clean","E":"Dirty"}),
    ("P05", "C", {"A":"Clean","B":"Dirty","C":"Clean","D":"Dirty","E":"Clean"}),
    ("P06", "B", {"A":"Dirty","B":"Clean","C":"Dirty","D":"Clean","E":"Dirty"}),
]

public_rows = []

for scenario_id, start, status_map in PUBLIC_SCENARIOS:
    world = run_episode_5(meu_agente_5, start, status_map)

    describe_world_5(world)

    public_rows.append({
        "Cenário": scenario_id,
        "Sucesso": world.is_clean(),
        "Passos": world.steps,
        "Ações inválidas": world.invalid_actions,
    })

public_results = pd.DataFrame(public_rows)

display(public_results)
print(f"Taxa de sucesso pública: {public_results['Sucesso'].mean():.1%}")
print(f"Média de passos: {public_results['Passos'].mean():.1f}")