import asyncio
from collections import defaultdict

from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from loguru import logger  # pyright: ignore[reportMissingImports]

from interaction_protocol.deliberation import extract_verdict


async def sync_delib(dilemma, clients, model_names, system_prompt_template, max_rounds=4, verbose=False):
    all_results = []
    histories = defaultdict(list)

    if verbose:
        logger.info("BEGINNING DELIBERATION")
        logger.info("Dilemma: {}", dilemma)

    for idx, name in enumerate(model_names, start=1):
        histories[name].extend([
            SystemMessage(content=system_prompt_template(agent=idx)),
            UserMessage(content=dilemma, source="Moderator")
        ])

    for round_idx in range(1, max_rounds + 1):
        coroutines = [clients[name].create(histories[name]) for name in model_names]
        results = await asyncio.gather(*coroutines)
        contents = [result.content for result in results]

        for name, content in zip(model_names, contents):
            message = AssistantMessage(content=content, source=name)
            histories[name].append(message)
            all_results.append(message)

        verdicts = [extract_verdict(content) for content in contents]

        if verbose:
            logger.info("=" * 50)
            logger.info("ROUND {} RESULTS", round_idx)
            logger.info("=" * 50)
            for name, content in zip(model_names, contents):
                logger.info("-" * 25)
                logger.info("{}", name)
                logger.info("{}", content)
                logger.info("-" * 25)

        if len(set(verdicts)) == 1:
            if verbose:
                logger.info("CONSENSUS, DELIBERATION COMPLETE")
            break
        else:
            if verbose:
                logger.info("NO CONSENSUS, PROCEEDING TO NEXT ROUND")
            agent_summaries = "\n".join(
                f"\nAgent {agent_idx} said:\n{content}"
                for agent_idx, content in enumerate(contents, start=1)
            )
            new_message_content = (
                f"Round {round_idx} Summary:\n"
                f"{agent_summaries}\n"
                f"\nConsensus was not reached. We proceed to Round {round_idx + 1}.\n"
            )
            update_message = UserMessage(content=new_message_content, source="Moderator")
            for history in histories.values():
                history.append(update_message)
    return all_results
