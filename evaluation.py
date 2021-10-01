import random
import main as mn

chars = 'qwertyuiopasdfghjklzxcvbnm     '


def test(epochs):
    probs = []
    correct_detections = []
    for x in range(epochs):
        word = ""
        for y in range(random.randint(8, 25)):
            word += random.choice(chars)
        prediction = mn.classify(word)
        if prediction:
            probs.append(prediction[0][1])
            # print(word)
        else:
            correct_detections.append(word)
    avg = sum(probs)/len(probs) * 100
    print(avg)
    print(len(correct_detections))
    return avg


print(mn.classify("wrogb worgwho"))
averages = []
for x in range(10):
    print("Epoch:", x+1)
    averages.append(test(1000))

# bad_responses = []
# bad_percentages = []
# random_sentences = """He drank life before spitting it out.
# There's an art to getting your way, and spitting olive pits across the table isn't it.
# It was a really good Monday for being a Saturday.
# The pet shop stocks everything you need to keep your anaconda happy.
# Grape jelly was leaking out the hole in the roof.
# He played the game as if his life depended on it and the truth was that it did.
# A suit of armor provides excellent sun protection on hot days.
# He was the only member of the club who didn't like plum pudding.
# After fighting off the alligator, Brian still had to face the anaconda.
# Siri became confused as we followed the detour and not her directions.
# The thunderous roar of the jet overhead confirmed her worst fears.
# The efficiency we have at removing trash has made creating trash more acceptable.
# His love of garlic ended once he lived down the street from the garlic processing plant.
# We will not allow you to bring your pet armadillo along.
# He wasn't bitter that she had moved on but from the radish.
# The hummingbird's wings blurred while it eagerly sipped the sugar water from the feeder.
# It's much more difficult to play tennis with a bowling ball than it is to bowl with a tennis ball.
# He embraced his new life as an eggplant.
# Joe discovered that traffic cones make excellent megaphones.
# No matter how beautiful the sunset, it saddened her knowing she was one day older.
# He found his art never progressed when he literally used his sweat and tears.
# The irony of the situation wasn't lost on anyone in the room.
# Just go ahead and press that button.
# It dawned on her that others could make her happier, but only she could make herself happy.
# Please put on these earmuffs because I can't you hear.
# She tilted her head back and let whip cream stream into her mouth while taking a bath.
# The best part of marriage is animal crackers with peanut butter.
# He is good at eating pickles and telling women about his emotional problems.
# It doesn't sound like that will ever be on my travel list.
# Written warnings in instruction manuals are worthless since rabbits can't read.
# With the high wind warning, it seemed like the perfect time to build a fire.
# The lake is a long way from here.
# Jim liked driving around town with his hazard lights on.
# Tomorrow will bring something new, so leave today as a memory.
# He wore the surgical mask in public not to keep from catching a virus, but to keep people away from him.
# Flesh-colored yoga pants were far worse than even he feared.
# Bill ran from the giraffe toward the dolphin.
# Karen believed all traffic laws should be obeyed except by herself.
# The old rusted farm equipment surrounded the house.
# He had decided to accept his fate of accepting his fate.
# He was sitting in a trash can with high street class.
# His mind was blown that there was nothing in space except space itself.
# He decided to fake his disappearance to avoid jail.
# Shakespeare was a famous 17th-century diesel mechanic.
# He decided that the time had come to be stronger than any of the excuses he'd used until then.
# He was an introvert that extroverts seemed to love.
# This is the last random sentence I will be writing and I am going to stop mid-sent
# The waves were crashing on the shore; it was a lovely sight.
# This made him feel like an old-style rootbeer float smells.
# Joyce enjoyed eating pancakes with ketchup.""".split('\n')
# for x in random_sentences:
#     if mn.classify(x):
#         bad_responses.append(1)
#         bad_percentages.append(mn.classify(x)[0][1])
#     else:
#         bad_responses.append(0)
#
# print(sum(bad_responses)/len(bad_responses) * 100)
# print(sum(bad_percentages)/len(bad_percentages) * 100)

sents = """
    You have to shut my students and staff. “Difficult” was struggling through a crowd, I rowed with, difficult to word for me. In the same as it since. But it lay untouched, case grew dusty as school. There’s less help. You have to drinking through a crowd, I spent all my early morning chromatic scales to shut me up to do this was school. There’s less help. You haven't playing them could match me in talent.
    Without my door to word for me. I struggled to manage. In their work with, difficult to ming alco
"""
for x in sents.split("."):
    print(mn.classify(x))
print(mn.classify("this apple is stupid"))