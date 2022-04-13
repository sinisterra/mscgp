# %%
import pykka
import time


class Evaluator(pykka.ThreadingActor):
    def on_receive(self, message):
        time.sleep(1)
        print(message)
        return tup


actor_ref = Evaluator.start()
acc = []
for e in (1, 2, 97, 98, 99):
    tup = ((("HIPERTENSION", e),), (("NEUMONIA", 1),))
    answer = actor_ref.ask(("Hola", tup))
    acc.append(answer)

print(acc)
