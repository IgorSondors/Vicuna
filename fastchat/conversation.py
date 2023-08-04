"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # The name of this template
    name: str
    # The System prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        """Convert the history to gradio chatbot format"""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# A template with one conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "What are the key differences between renewable and non-renewable energy sources?",
            ),
            (
                "Assistant",
                "Renewable energy sources are those that can be replenished naturally in a relatively "
                "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
                "Non-renewable energy sources, on the other hand, are finite and will eventually be "
                "depleted, such as coal, oil, and natural gas. Here are some key differences between "
                "renewable and non-renewable energy sources:\n"
                "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
                "energy sources are finite and will eventually run out.\n"
                "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
                "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
                "and other negative effects.\n"
                "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
                "have lower operational costs than non-renewable sources.\n"
                "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
                "locations than non-renewable sources.\n"
                "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
                "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
                "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
                "non-renewable sources are not, and their depletion can lead to economic and social instability.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# My template planshet
register_conv_template(
    Conversation(
        name="planshet",
        system="A chat between a human and an artificial intelligence Named Entities Recognition Assistant. Answer of the system has exactly template and no more other words. The system is double checking the answer before give that."
        "The system can only parse named entities like Brand name, Device name, Color from text and nothing else. That is the template of the answer: 'Brand:___\nDevice:___\nColor:___'."
        "Here are some axamples of 'Device name' field: TN1002T, T700i, TB04, Iconia Tab A100, Novo 10 Hero, T20, Iplay 20p, iPad 16Gb Wi-Fi, 101 G9 16gb, AlLdocube T1009, AP-701, Eee Pad Transformer Slider SL101, B10, FirstTab TP-17, PW9772I, 7056G, BeTAB 7071, PlayBook 16Gb, Tab 10, Pad C7.3b, i7-Stylus U107GT-S, V-T500-E, Hi10, Kyros MID1060, ZiiO 10 8Gb, B700, Ursus 10EV 3G, DTR-07FSBH, Streak Mini, iD 7, V99x-16, ECO 7.0 QC, CityNight, TM105A, G37, ETL-T711, eTab A07-0104, F-TP1009, P-700, informer 901, A1 3G, IQ 300, Stylistic Q550, Happy-01, PC-1031, Galapad 7, GS700, S1082, QV1030, GT-X831, GU707W, MagicPad L701W, TAB A73, Smart 7 3G, Slate 500, Flyer, D71-B, Alpha Tab, Pad V6, Ideos S7 Slim, H-SP9701G, U7i, PX-0705, TH73, EZpad 8, Manshet KS-122, K-88, 737, TL801W4, Skybox 709, V900 Optimus Pad, IdeaPad K1-10W32K, A711, Tablet Master 2, LT1001, MD-001H, M10, Enjoy 10, MID900, ePad 1002, MID8125, Surface 32Gb Touch Cover, Miowork A335, XOOM Wi-Fi, TPC-10Y, MID-721, Neo 9.7, T737 PRO, Lumia 2520, On-Tab 7011, S703, TPC-10, Cosmo, V812, Tablet TW117, MyPad D7G1, T703, T10, Lucid Tablet, 712W, PI3100, M2 3G, SURFpad 4 L, Mobii TEGRA Tablet 10.1, MultiPad PMP3384B, Evolution Note-700, Flame 16GB, Pad 10.4, CiTiPAD L-700, 7001, RMD-700, RTB 8.4 Joy, Luna RMD-10, RoverPad 3W A73, P12, Galaxy Tab A7 Lite 32GB, ST-TPC0701, Smart TG715, Maximus T1001, M722-A13, 3GD52i, Tablet S 16Gb, Andromeda S900, Sky 7 E200 3G, SD700, Tab 10S, P80 4G, TF-MID701G, Effect 10.1 3G, TM-7010, Tablet A10, PAD T07R, Libretto W100-106, Brevis 701WA, 1000, Tab 4G 8-1, ViewPad 10, TAB 8001, MiPad 2 16Gb, TAB07-200, V9A, Vivid ZT-10003G, Pad Master 10.1, ITQ701, NetTAB Rune, JR70, SHIELD Tablet 16Gb Wi-Fi, Sinapse T1000 MID, M2 3G, 1055, V9+",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Планшетный компьютер Digma Optima Prime 4 3G Black",
            ),
            (
                "Assistant",
                "Brand name:Digma\nDevice:Optima Prime 4 3G\nColor:Black.",
            ),

            (
                "Human",
                "Планшет SunWind Sky 1262C 3G Sc7731e 4C RAM2Gb ROM32Gb 10.1 IPS 1280x800 3G Android 11.0 Go черный",
            ),
            (
                "Assistant",
                "Brand name:SunWind\nDevice:Sky 1262C 3G\nColor:черный.",
            ),

            (
                "Human",
                "Планшет Apple iPad mini Retina 16Gb Wi-Fi Silver (ME279RU/A)",
            ),
            (
                "Assistant",
                "Brand name:Apple\nDevice:iPad mini with Retina display Wi-Fi 16GB\nColor:Silver.",
            ),
            
            (
                "Human",
                "Планшет Samsung Galaxy Tab S6 10.5 SM-T860 128Gb Wi-Fi (голубой)",
            ),
            (
                "Assistant",
                "Brand name:Samsung\nDevice:Galaxy Tab S6 10.5 128Gb\nColor:голубой.",
            ),

        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# My template planshet
register_conv_template(
    Conversation(
        name="planshet_without_device_examples",
        system="A chat between a human and an artificial intelligence Named Entities Recognition Assistant. Answer of the system has exactly template and no more other words. The system is double checking the answer before give that."
        "The system can only parse named entities like Brand name, Device name, Color from text and nothing else. That is the template of the answer: 'Brand:___\nDevice:___\nColor:___'.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Планшетный компьютер Digma Optima Prime 4 3G Black",
            ),
            (
                "Assistant",
                "Brand name:Digma\nDevice:Optima Prime 4 3G\nColor:Black.",
            ),

            (
                "Human",
                "Планшет SunWind Sky 1262C 3G Sc7731e 4C RAM2Gb ROM32Gb 10.1 IPS 1280x800 3G Android 11.0 Go черный",
            ),
            (
                "Assistant",
                "Brand name:SunWind\nDevice:Sky 1262C 3G\nColor:черный.",
            ),

            (
                "Human",
                "Планшет Apple iPad mini Retina 16Gb Wi-Fi Silver (ME279RU/A)",
            ),
            (
                "Assistant",
                "Brand name:Apple\nDevice:iPad mini with Retina display Wi-Fi 16GB\nColor:Silver.",
            ),
            
            (
                "Human",
                "Планшет Samsung Galaxy Tab S6 10.5 SM-T860 128Gb Wi-Fi (голубой)",
            ),
            (
                "Assistant",
                "Brand name:Samsung\nDevice:Galaxy Tab S6 10.5 128Gb\nColor:голубой.",
            ),

        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# My template planshet
register_conv_template(
    Conversation(
        name="planshet_big",
        system="A chat between a human and an artificial intelligence Named Entities Recognition Assistant. Answer of the system has exactly template and no more other words. The system is double checking the answer before give that."
        "The system can only parse named entities like Brand, Device from text and nothing else. That is the template of the answer: 'Brand:___\nDevice:___\nEND_OF_ANSWER'.",
        roles=("Human", "Assistant"),
        messages=(
            
			(
				"Human",
				"Планшетный компьютер 7\" Alcatel i216X Pixi 7 1.2ГГц/512Mb/4Gb/7\" 960*540/WIFI/3G/Bluetooth/GPS/Android 4.4",
			),
			(
				"Assistant",
				"Brand:Alcatel\nDevice:Pixi 7\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Apple iPad 4 Retina 16Gb Wi-Fi Black (MD510TU/A MD510RS/A)",
			),
			(
				"Assistant",
				"Brand:Apple\nDevice:iPad 4 16Gb Wi-Fi\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Achos 101e Neon 16GB",
			),
			(
				"Assistant",
				"Brand:Achos\nDevice:101e Neon\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Wexler Планшетный компьютер 7\" WEXLER.TAB 7iS 8GB 3G black Cortex A9 1.5Ггц/1Гб/8Гб/7\" 1280*800/WIFI/Bluetooth/3G/GPS/Android 4.1",
			),
			(
				"Assistant",
				"Brand:WEXLER\nDevice:TAB 7iS\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 CS1219PL Планшет DIGMA CITI Octa 10, 4GB, 64GB, 3G, 4G, Android 9.0 черный",
			),
			(
				"Assistant",
				"Brand:DIGMA\nDevice:CITI Octa 10\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 53013NAH Планшет MATEPAD SE 4GB WIFI 64GB AGS5-W09 BLACK HUAWEI",
			),
			(
				"Assistant",
				"Brand:HUAWEI\nDevice:MATEPAD SE\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Microsoft Surface Pro X MSQ2 16/512Gb Platinum",
			),
			(
				"Assistant",
				"Brand:Microsoft\nDevice:Surface Pro X MSQ2\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Prestigio Grace 5588 16 Гб черный",
			),
			(
				"Assistant",
				"Brand:Prestigio\nDevice:Grace 5588\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Xiaomi Mi Pad 4 8\" 1920x1200 IPS, Snapdragon 660, 4Gb RAM, 64Gb, WiFi, BT, 6000mAh, Android 8.1, черный",
			),
			(
				"Assistant",
				"Brand:Xiaomi\nDevice:Mi Pad 4 \nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Xiaomi MiPad 4 32Gb Wi-Fi black",
			),
			(
				"Assistant",
				"Brand:Xiaomi\nDevice:MiPad 4 32Gb\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 53013BAH Планшет Huawei MatePad T10s AGS3K-L09, 4GB, 64GB, 3G, 4G, HarmonyOS 2 темно-синий",
			),
			(
				"Assistant",
				"Brand:Huawei\nDevice:MatePad T10s\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Sony Xperia Tablet Z 32Gb White (Android 4.1/APQ8064 1500Mhz/10.1\" (1920x1200)/2048Mb/32Gb/WiFi/BT) [SGP-312RU]",
			),
			(
				"Assistant",
				"Brand:Sony\nDevice:Xperia Tablet Z\nEND_OF_ANSWER",
			),
			

        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# My template planshet
register_conv_template(
    Conversation(
        name="planshet_big2",
        system="A chat between a human and an artificial intelligence Named Entities Recognition Assistant. Answer of the system has exactly template and no more other words. The system is double checking the answer before give that."
        "The system can only parse named entities like Brand name, Device name from text and nothing else. That is the template of the answer: 'Brand:___\nDevice:___\nEND_OF_ANSWER'.",
        roles=("Human", "Assistant"),
        messages=(
            
			(
				"Human",
				"Планшетный компьютер 7\" Alcatel i216X Pixi 7 1.2ГГц/512Mb/4Gb/7\" 960*540/WIFI/3G/Bluetooth/GPS/Android 4.4",
			),
			(
				"Assistant",
				"Brand:Alcatel\nDevice:Pixi 7\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Apple iPad 4 Retina 16Gb Wi-Fi Black (MD510TU/A MD510RS/A)",
			),
			(
				"Assistant",
				"Brand:Apple\nDevice:iPad 4 16Gb Wi-Fi\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Achos 101e Neon 16GB",
			),
			(
				"Assistant",
				"Brand:Archos\nDevice:101e Neon\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Wexler Планшетный компьютер 7\" WEXLER.TAB 7iS 8GB 3G black Cortex A9 1.5Ггц/1Гб/8Гб/7\" 1280*800/WIFI/Bluetooth/3G/GPS/Android 4.1",
			),
			(
				"Assistant",
				"Brand:Wexler\nDevice:TAB 7iS\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 CS1219PL Планшет DIGMA CITI Octa 10, 4GB, 64GB, 3G, 4G, Android 9.0 черный",
			),
			(
				"Assistant",
				"Brand:Digma\nDevice:CITI Octa 10\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 53013NAH Планшет MATEPAD SE 4GB WIFI 64GB AGS5-W09 BLACK HUAWEI",
			),
			(
				"Assistant",
				"Brand:Huawei\nDevice:MATEPAD SE 64Gb\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Microsoft Surface Pro X MSQ2 16/512Gb Platinum",
			),
			(
				"Assistant",
				"Brand:Microsoft\nDevice:Surface Pro X MSQ2\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Prestigio Grace 5588 16 Гб черный",
			),
			(
				"Assistant",
				"Brand:Prestigio\nDevice:Grace 5588\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Xiaomi Mi Pad 4 8\" 1920x1200 IPS, Snapdragon 660, 4Gb RAM, 64Gb, WiFi, BT, 6000mAh, Android 8.1, черный",
			),
			(
				"Assistant",
				"Brand:Xiaomi\nDevice:Mi Pad 4 64Gb\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Xiaomi MiPad 4 32Gb Wi-Fi black",
			),
			(
				"Assistant",
				"Brand:Xiaomi\nDevice:Mi Pad 4 32Gb\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 53013BAH Планшет Huawei MatePad T10s AGS3K-L09, 4GB, 64GB, 3G, 4G, HarmonyOS 2 темно-синий",
			),
			(
				"Assistant",
				"Brand:Huawei\nDevice:MatePad T10S\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Sony Xperia Tablet Z 32Gb White (Android 4.1/APQ8064 1500Mhz/10.1\" (1920x1200)/2048Mb/32Gb/WiFi/BT) [SGP-312RU]",
			),
			(
				"Assistant",
				"Brand:Sony\nDevice:Xperia Tablet Z\nEND_OF_ANSWER",
			),

			(
				"Human",
				"Планшет Prestigio MultiPad Color 8.0 3G (PMT5887)",
			),
			(
				"Assistant",
				"Brand:Prestigio\nDevice:MultiPad Color 8.0 3G\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет BQ 1081G White 3G/10.1\"/1024*600/microSD/1Gb/8Gb/4*1ГГц/And7.0/GPS/4000мАч",
			),
			(
				"Assistant",
				"Brand:BQ\nDevice:1081g\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Asus MeMO Pad HD 7 ME173X 1Gb 16Gb 7 BT Cam GPS 3950мАч Android 4.2 90NK00B4-M04710 Желтый",
			),
			(
				"Assistant",
				"Brand:Asus\nDevice:MeMO Pad HD 7 ME173X\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Ginzzu GT-8010 Black",
			),
			(
				"Assistant",
				"Brand:Ginzzu\nDevice:GT-8010\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Blackview Tab 10 4/64 Silver с клавиатурой",
			),
			(
				"Assistant",
				"Brand:Blackview\nDevice:Tab 10\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Realme Pad RMP2103 Gray (MediaTek Helio G80 2.0 GHz/4096Mb/64Gb/LTE/Wi-Fi/Bluetooth/Cam/10.4/2000x1200/Android)",
			),
			(
				"Assistant",
				"Brand:Realme\nDevice:Pad 10.4\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Acer ICONIA B1-710 16GB MT8317T/1Gb/16GB/7\"(1024x600)/WiFi/BT/GPS/Android 4.2",
			),
			(
				"Assistant",
				"Brand:Acer\nDevice:Iconia Tab B1-710\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Fujitsu STYLISTIC M532 32Gb 3G Tegra 3/1GB/10,1\" 1280x800/Cam/MicroSD/Wi-Fi/GPS/BT/Android 4.0",
			),
			(
				"Assistant",
				"Brand:Fujitsu\nDevice:Stylistic M532\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет JUMPER Ezpad 8 4GB 64GB Windows 10 Home серый",
			),
			(
				"Assistant",
				"Brand:Jumper\nDevice:EZpad 8\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Samsung Galaxy Note 8.0 16Gb GT-N5100 White (Android 4.1/Exynos 4412 1600Mhz/8\" (1280x800)/2048Mb/16Gb/3G/WiFi/BT) [GT-N5100ZWAMGF]",
			),
			(
				"Assistant",
				"Brand:Samsung\nDevice:Galaxy Note 8.0\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Huawei MediaPad T1 8.0 8Gb S8-701U (Qualcomm MSM8212 1.2 GHz/1024Mb/8Gb/Wi-Fi/3G/Bluetooth/GPS/Webcam/8.0/1280x800/Android)",
			),
			(
				"Assistant",
				"Brand:Huawei\nDevice:MediaPad T1 8.0\nEND_OF_ANSWER",
			),
					

        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# My template planshet
register_conv_template(
    Conversation(
        name="planshet_small",
        system="A chat between a human and an artificial intelligence Named Entities Recognition Assistant. Answer of the system has exactly template and no more other words. The system is double checking the answer before give that."
        "The system can only parse named entities like Brand name, Device name from text and nothing else. That is the template of the answer: 'Brand:___\nDevice:___\nEND_OF_ANSWER'.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Планшетный компьютер Digma Optima Prime 4 3G Black",
            ),
            (
                "Assistant",
                "Brand name:Digma\nDevice:Optima Prime 4 3G\nEND_OF_ANSWER",
            ),

            (
                "Human",
                "Планшет SunWind Sky 1262C 3G Sc7731e 4C RAM2Gb ROM32Gb 10.1 IPS 1280x800 3G Android 11.0 Go черный",
            ),
            (
                "Assistant",
                "Brand name:SunWind\nDevice:Sky 1262C 3G\nEND_OF_ANSWER",
            ),

            (
                "Human",
                "Планшет Apple iPad mini Retina 16Gb Wi-Fi Silver (ME279RU/A)",
            ),
            (
                "Assistant",
                "Brand name:Apple\nDevice:iPad mini with Retina display Wi-Fi 16GB\nEND_OF_ANSWER",
            ),
            
            (
                "Human",
                "Планшет Samsung Galaxy Tab S6 10.5 SM-T860 128Gb Wi-Fi (голубой)",
            ),
            (
                "Assistant",
                "Brand name:Samsung\nDevice:Galaxy Tab S6 10.5 128Gb\nEND_OF_ANSWER",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)


# My template planshet
register_conv_template(
    Conversation(
        name="obuv",
        system="A chat between a human and an artificial intelligence Named Entities Recognition Assistant. Answer of the system has exactly template and no more other words. The system is double checking the answer before give that."
        "The system can only parse named entities like Brand, Product from text and nothing else. That is the template of the answer: 'Brand:___\nProduct:___\nEND_OF_ANSWER'.",
        roles=("Human", "Assistant"),
        messages=(
            
			(
				"Human",
				"Планшетный компьютер 7\" Alcatel i216X Pixi 7 1.2ГГц/512Mb/4Gb/7\" 960*540/WIFI/3G/Bluetooth/GPS/Android 4.4",
			),
			(
				"Assistant",
				"Brand:Alcatel\nProduct:Pixi 7\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Apple iPad 4 Retina 16Gb Wi-Fi Black (MD510TU/A MD510RS/A)",
			),
			(
				"Assistant",
				"Brand:Apple\nProduct:iPad 4 16Gb Wi-Fi\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Achos 101e Neon 16GB",
			),
			(
				"Assistant",
				"Brand:Achos\nProduct:101e Neon\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Wexler Планшетный компьютер 7\" WEXLER.TAB 7iS 8GB 3G black Cortex A9 1.5Ггц/1Гб/8Гб/7\" 1280*800/WIFI/Bluetooth/3G/GPS/Android 4.1",
			),
			(
				"Assistant",
				"Brand:WEXLER\nProduct:TAB 7iS\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 CS1219PL Планшет DIGMA CITI Octa 10, 4GB, 64GB, 3G, 4G, Android 9.0 черный",
			),
			(
				"Assistant",
				"Brand:DIGMA\nProduct:CITI Octa 10\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 53013NAH Планшет MATEPAD SE 4GB WIFI 64GB AGS5-W09 BLACK HUAWEI",
			),
			(
				"Assistant",
				"Brand:HUAWEI\nProduct:MATEPAD SE\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Microsoft Surface Pro X MSQ2 16/512Gb Platinum",
			),
			(
				"Assistant",
				"Brand:Microsoft\nProduct:Surface Pro X MSQ2\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Prestigio Grace 5588 16 Гб черный",
			),
			(
				"Assistant",
				"Brand:Prestigio\nProduct:Grace 5588\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Xiaomi Mi Pad 4 8\" 1920x1200 IPS, Snapdragon 660, 4Gb RAM, 64Gb, WiFi, BT, 6000mAh, Android 8.1, черный",
			),
			(
				"Assistant",
				"Brand:Xiaomi\nProduct:Mi Pad 4 \nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Xiaomi MiPad 4 32Gb Wi-Fi black",
			),
			(
				"Assistant",
				"Brand:Xiaomi\nProduct:MiPad 4 32Gb\nEND_OF_ANSWER",
			),
			(
				"Human",
				"934 53013BAH Планшет Huawei MatePad T10s AGS3K-L09, 4GB, 64GB, 3G, 4G, HarmonyOS 2 темно-синий",
			),
			(
				"Assistant",
				"Brand:Huawei\nProduct:MatePad T10s\nEND_OF_ANSWER",
			),
			(
				"Human",
				"Планшет Sony Xperia Tablet Z 32Gb White (Android 4.1/APQ8064 1500Mhz/10.1\" (1920x1200)/2048Mb/32Gb/WiFi/BT) [SGP-312RU]",
			),
			(
				"Assistant",
				"Brand:Sony\nProduct:Xperia Tablet Z\nEND_OF_ANSWER",
			),
			

        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)


# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)


# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        system="",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt",
        system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.
""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        name="bard",
        system="",
        roles=("0", "1"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        system="",
        roles=("<human>", "<bot>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        system="",
        roles=("<|prompt|>", "<|answer|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)


if __name__ == "__main__":
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
