

class Ticket:

    def __init__(self, kid, adult, day):
        self.kid = kid
        self.adult = adult
        self.day = day

    def prices(self):
        if self.day in range(6):
            print(100 * self.adult + 50 * self.kid)
        else:
            print(1.2 * (100 * self.adult + 50 * self.kid))

m = Ticket(1, 2, 1)
m.prices()
