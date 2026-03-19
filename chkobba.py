from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUITS: Tuple[str, ...] = ("Denari", "Sbebit", "Kloub", "Chjer")
FACE_NAMES: Dict[int, str] = {8: "Jack", 9: "Knight", 10: "King"}
INITIAL_TABLE_CARDS = 4
CARDS_PER_DEAL = 3
TOTAL_CARDS = 40


@dataclass(frozen=True, order=True)
class Card:
    """Represents a single Tunisian Chkobba card."""

    suit: str
    value: int

    def __post_init__(self) -> None:
        if self.suit not in SUITS:
            raise ValueError(f"Unsupported suit: {self.suit}")
        if self.value < 1 or self.value > 10:
            raise ValueError(f"Unsupported value: {self.value}")

    @property
    def label(self) -> str:
        face_name = FACE_NAMES.get(self.value, str(self.value))
        return f"{face_name} of {self.suit}"

    def __str__(self) -> str:
        return self.label


class Deck:
    """40-card Tunisian Chkobba deck."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._random = random.Random(seed)
        self.cards: List[Card] = [Card(suit=suit, value=value) for suit in SUITS for value in range(1, 11)]
        self._random.shuffle(self.cards)

    def draw(self, count: int = 1) -> List[Card]:
        if count < 0:
            raise ValueError("Cannot draw a negative number of cards")
        if count > len(self.cards):
            raise ValueError("Not enough cards left in the deck")
        drawn = self.cards[:count]
        del self.cards[:count]
        return drawn

    def __len__(self) -> int:
        return len(self.cards)


@dataclass
class Player:
    """Tracks each player's hand, captures, and scopa count."""

    name: str
    hand: List[Card] = field(default_factory=list)
    captured_cards: List[Card] = field(default_factory=list)
    scopa_count: int = 0

    def receive_cards(self, cards: Iterable[Card]) -> None:
        self.hand.extend(cards)

    def play_card(self, index: int) -> Card:
        if index < 0 or index >= len(self.hand):
            raise IndexError(f"Invalid card index {index} for player {self.name}")
        return self.hand.pop(index)


@dataclass
class TurnResult:
    player_name: str
    played_card: Card
    captured_cards: List[Card]
    scopa_scored: bool
    capture_reason: str
    table_after_turn: List[Card]


class Game:
    """Core Tunisian Chkobba engine for a two-player round."""

    def __init__(self, player_names: Sequence[str], seed: Optional[int] = None) -> None:
        if len(player_names) != 2:
            raise ValueError("This MVP supports exactly 2 players")

        self._random = random.Random(seed)
        self.deck = Deck(seed=seed)
        self.players = [Player(name=name) for name in player_names]
        self.table_cards: List[Card] = []
        self.current_player_index = 0
        self.last_capturing_player: Optional[Player] = None
        self.turn_history: List[TurnResult] = []
        self._round_finished = False

        self._initial_deal()

    def _initial_deal(self) -> None:
        self.table_cards.extend(self.deck.draw(INITIAL_TABLE_CARDS))
        for player in self.players:
            player.receive_cards(self.deck.draw(CARDS_PER_DEAL))

    def _deal_if_needed(self) -> None:
        if self.deck and all(not player.hand for player in self.players):
            for player in self.players:
                player.receive_cards(self.deck.draw(CARDS_PER_DEAL))

    def is_round_over(self) -> bool:
        return self._round_finished

    def available_capture_options(self, played_card: Card) -> Tuple[str, List[List[Card]]]:
        """
        Apply the mandatory Tunisian priority rules.

        Priority 1: any exact same-value table cards MUST be captured.
        Priority 2: only when there is no exact match, sum-combinations can be captured.
        """
        exact_matches = [card for card in self.table_cards if card.value == played_card.value]
        if exact_matches:
            return "exact", [exact_matches]

        sum_matches: List[List[Card]] = []
        for combo in self._all_table_combinations():
            if sum(card.value for card in combo) == played_card.value:
                sum_matches.append(list(combo))
        return "sum", sum_matches

    def _all_table_combinations(self) -> Iterable[Tuple[Card, ...]]:
        for size in range(1, len(self.table_cards) + 1):
            yield from combinations(self.table_cards, size)

    def play_turn(self, hand_index: int, capture_cards: Optional[Sequence[Card]] = None) -> TurnResult:
        if self._round_finished:
            raise RuntimeError("The round is already over")

        player = self.players[self.current_player_index]
        played_card = player.play_card(hand_index)
        capture_mode, options = self.available_capture_options(played_card)
        captured_cards: List[Card] = []
        scopa_scored = False
        capture_reason = "drop"

        if options:
            if capture_mode == "exact":
                # Exact match is mandatory and overrides any possible sum capture.
                captured_cards = options[0]
                capture_reason = "exact match"
            else:
                if capture_cards is None:
                    raise ValueError("A sum capture choice is required when multiple combinations exist")
                validated_capture = self._validate_sum_capture_choice(played_card, capture_cards, options)
                captured_cards = validated_capture
                capture_reason = "sum match"

            for card in captured_cards:
                self.table_cards.remove(card)
            player.captured_cards.extend([played_card, *captured_cards])
            self.last_capturing_player = player

            # A sweep counts only if it empties the table and the played card is not the round's final card.
            if not self.table_cards and not self._is_final_card_of_round(player):
                player.scopa_count += 1
                scopa_scored = True
        else:
            self.table_cards.append(played_card)

        self.turn_history.append(
            TurnResult(
                player_name=player.name,
                played_card=played_card,
                captured_cards=list(captured_cards),
                scopa_scored=scopa_scored,
                capture_reason=capture_reason,
                table_after_turn=list(self.table_cards),
            )
        )

        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self._deal_if_needed()
        self._check_round_end()
        return self.turn_history[-1]

    def _validate_sum_capture_choice(
        self,
        played_card: Card,
        capture_cards: Sequence[Card],
        valid_options: Sequence[Sequence[Card]],
    ) -> List[Card]:
        chosen = list(capture_cards)
        if not chosen:
            raise ValueError("Chosen sum capture cannot be empty")
        if sum(card.value for card in chosen) != played_card.value:
            raise ValueError("Chosen sum capture does not equal the played card value")

        if any(card not in self.table_cards for card in chosen):
            raise ValueError("Chosen sum capture contains cards not present on the table")

        chosen_signature = sorted(chosen)
        valid_signatures = [sorted(option) for option in valid_options]
        if chosen_signature not in valid_signatures:
            raise ValueError("Chosen sum capture is not a valid available combination")
        return chosen

    def _is_final_card_of_round(self, player_who_played: Player) -> bool:
        return len(self.deck) == 0 and all(not player.hand for player in self.players if player is not player_who_played) and not player_who_played.hand

    def _check_round_end(self) -> None:
        if len(self.deck) == 0 and all(not player.hand for player in self.players):
            if self.table_cards and self.last_capturing_player is not None:
                self.last_capturing_player.captured_cards.extend(self.table_cards)
                self.table_cards.clear()
            self._round_finished = True

    def choose_random_sum_capture(self, played_card: Card) -> Optional[List[Card]]:
        capture_mode, options = self.available_capture_options(played_card)
        if capture_mode == "sum" and options:
            return self._random.choice(options)
        return None

    def simulate_random_round(self) -> Dict[str, Dict[str, int]]:
        while not self.is_round_over():
            player = self.players[self.current_player_index]
            hand_index = self._random.randrange(len(player.hand))
            preview_card = player.hand[hand_index]
            capture_mode, options = self.available_capture_options(preview_card)
            capture_choice = None
            if capture_mode == "sum" and options:
                capture_choice = self._random.choice(options)
            self.play_turn(hand_index=hand_index, capture_cards=capture_choice)
        return self.calculate_scores()

    def calculate_scores(self) -> Dict[str, Dict[str, int]]:
        if not self._round_finished:
            raise RuntimeError("Cannot score a round before all cards are played")

        breakdown = {
            player.name: {
                "cards": 0,
                "denari": 0,
                "sette_bello": 0,
                "primiera": 0,
                "scopa": player.scopa_count,
                "total": player.scopa_count,
            }
            for player in self.players
        }

        self._award_majority_point(
            breakdown,
            category="cards",
            counts={player.name: len(player.captured_cards) for player in self.players},
            threshold=20,
        )
        self._award_majority_point(
            breakdown,
            category="denari",
            counts={player.name: self._count_suit(player.captured_cards, "Denari") for player in self.players},
            threshold=5,
        )
        self._award_sette_bello(breakdown)
        self._award_primiera(breakdown)

        for player in self.players:
            breakdown[player.name]["total"] = sum(
                value for key, value in breakdown[player.name].items() if key != "total"
            )
        return breakdown

    def _award_majority_point(
        self,
        breakdown: Dict[str, Dict[str, int]],
        category: str,
        counts: Dict[str, int],
        threshold: int,
    ) -> None:
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        leader_name, leader_count = sorted_counts[0]
        _, second_count = sorted_counts[1]
        if leader_count > threshold and leader_count > second_count:
            breakdown[leader_name][category] = 1

    def _award_sette_bello(self, breakdown: Dict[str, Dict[str, int]]) -> None:
        for player in self.players:
            if any(card.suit == "Denari" and card.value == 7 for card in player.captured_cards):
                breakdown[player.name]["sette_bello"] = 1
                return

    def _award_primiera(self, breakdown: Dict[str, Dict[str, int]]) -> None:
        seven_counts = {player.name: self._count_value(player.captured_cards, 7) for player in self.players}
        six_counts = {player.name: self._count_value(player.captured_cards, 6) for player in self.players}

        sorted_sevens = sorted(seven_counts.items(), key=lambda item: item[1], reverse=True)
        leader_name, leader_sevens = sorted_sevens[0]
        _, second_sevens = sorted_sevens[1]
        if leader_sevens > 2 and leader_sevens > second_sevens:
            breakdown[leader_name]["primiera"] = 1
            return

        if leader_sevens == second_sevens == 2:
            sorted_sixes = sorted(six_counts.items(), key=lambda item: item[1], reverse=True)
            six_leader_name, six_leader_count = sorted_sixes[0]
            _, six_second_count = sorted_sixes[1]
            if six_leader_count > 2 and six_leader_count > six_second_count:
                breakdown[six_leader_name]["primiera"] = 1

    @staticmethod
    def _count_suit(cards: Sequence[Card], suit: str) -> int:
        return sum(1 for card in cards if card.suit == suit)

    @staticmethod
    def _count_value(cards: Sequence[Card], value: int) -> int:
        return sum(1 for card in cards if card.value == value)

    def scoring_breakdown_lines(self, scores: Dict[str, Dict[str, int]]) -> List[str]:
        lines: List[str] = []
        for player in self.players:
            stats = scores[player.name]
            lines.append(
                (
                    f"{player.name}: total={stats['total']} | cards={stats['cards']} | denari={stats['denari']} | "
                    f"sette_bello={stats['sette_bello']} | primiera={stats['primiera']} | scopa={stats['scopa']} | "
                    f"captured={len(player.captured_cards)}"
                )
            )
        return lines


if __name__ == "__main__":
    seed = 7
    print(f"Starting automated Tunisian Chkobba simulation with seed={seed}")
    game = Game(player_names=("Player 1", "Player 2"), seed=seed)
    final_scores = game.simulate_random_round()

    print("\nTurn summary:")
    for turn_number, turn in enumerate(game.turn_history, start=1):
        captured = ", ".join(str(card) for card in turn.captured_cards) or "None"
        table_state = ", ".join(str(card) for card in turn.table_after_turn) or "Empty"
        scopa_note = " +SCOPA" if turn.scopa_scored else ""
        print(
            f"Turn {turn_number:02d} | {turn.player_name} played {turn.played_card} | "
            f"{turn.capture_reason} | captured: {captured}{scopa_note} | table: {table_state}"
        )

    print("\nFinal scoring breakdown:")
    for line in game.scoring_breakdown_lines(final_scores):
        print(line)
