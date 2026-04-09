#!/usr/bin/env python3
"""
Bug #0000: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['branch', 'fpn_branch'], ['fusion', 'conv_bn_selu'], ['fusion', 'conv_tanh_head'], ['layout', 'resize_linear_aligncorners'], ['constant', 'relu_residual_sum'], ['normalization', 'reduce_l2_last']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0000.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0000.onnx")
INPUT = np.frombuffer(bytes.fromhex("6f2f29bfc46724bf99040ebfd679abbf5a4d72bf3f71463f577707becb7f093fd3209c3e53a64bbf91ff103e62b50e3e84ce72be05e3343f5df5ea3eb005c4bf2488eabdcfc908c047eed73da06c003f2d22893f18ab853a10ecc5bf71e2ec3f64013f3f6173d13e52da8dbf9930e8bde404afbffd5544bfe7a4e93f96c6a93fd40d5a3f5e270dbf3bcdc4bed0032fbe346d0c3fae069bbdc91cbbbe196a0f400e1b883f741dc33d9dae843fc95dabbf9158ae3fd743223f3715223fd480d8bf6c04b23e2b09f2bfbe60683fb659e9bf00df943eb02a3e3f21eceb3e9c3ae4be6bf0b93e581812bf93e805bf077b8ebed0352cbeae8ad63e13b85cbed9bdb6be8305fabf344ca03eecec83be0a5d363f6e34eebdc8e527bf4b7bad3f532f353ece1837be6a4b8b3fc3eafe3f3250ebbe9213ad3fa62f1bbf3f69b83f3978c2beba4c17becdd3913fe862e1bf2f6d8dbffc7506bc686f9e3d520d1a3f8b8d00bffb4f093c22c033bf13114c3fde9f50bf9b85b23fd6419fbf372d12be25c8513fe9358d3f698e8d3dff7f21bf6877383f9ec6333ec6a3c53f8c2f9cbebd265dbe335ec9bfeec95ebdb92756bea57325be4245143fc7dab9bfdca1ebbe6b89e7be4642cdbf2cdf19408424e93ee54f233f17e5963ec5edfdbb9c9d303f6ad8bdbfa22b30be7aa3a8bf572db13fc229f6bf3f37f3bf48584a3fa1b408bfd30d1b3f79674b3f6e6be8be8d125a3f4afe513fa8840d3f9bfa97be4f6828c0a6f1e03f0378093f4dae843fac7ceb3e3af862be41a12abfa906863c207cc53f136958beec27c4bf32967f3f57b8173d75bcaf3e687756bf735c6ebea8ce3cbfe03ca4bed4bb8e3f846be9bfd6ce293f5e3414bf411cb43fc964f3bec60ca6bf6622c63f684862bf4fb07dbf5e515abf19dcdabf4e9104bfce8b233f7fb1853e78f980be23d3ac3fe466dabf0d06193e4eeb463f0c8303401ad8bdbf486488bfccbc683fc0594c3f7929063ffa10afbf30bd8c3f237f7b3c605697bd9959cdbedc7ec43ffc0c10bda32d4fbfc2cb1d40a106e93f8a85343ff0e9193fc338ff3fe1f618bf2196adbf198b13bf447689bfeb47e53eda4bb23dd3b92cbf80116c3fbb954b3fe1f396bebb35d23e8519bebd68ab9c3e7ff0293ef8ba1b407f40143db4955f3e6bd1c1be1276b2bf9991fa3f543283bfda91fabdd918b63d9ec632be00db99bf31755fbff4094bbf10f4893e0767103ea1dd513fab7bca3e43a6e43f5b2d3b40f9176dbf587f963fe81be2be41383ebf6a2d703f8b098ebfba3d3bbf74b893be10eb3b40f5b5ed3ea44457bff471fa3be191823fc5dcd3be51309b3e7683123fede24dbfc5bb9ebbebbfe63ee982b7bf75a7183ef5999abf0696d6bea4764fbfe3e7b83e871e2f3e2a759b3f7cc4ac3fa0f3133dd198ea3eab188cbfb618c83f6260d83f1669a9bf36c8babf3c9400406e68c5bf66d772bffff0823e4181a23fb1c44dbe61780fbeca56bfbd5c3927bf724f03be747831bf1e4aad3fb27d983e251e38bf697e15bf064cdf3eb3f5143e286dfd3eea73833f4138d43d7f29bbbf891dc13e9c6bafbf2b9351bf4100943eca7704be7e28fdbd6239463eef1b323f714d3b3e31f8893f1abadd3d6d56d13e9879e13f1683a03fc38c56be2e07373f6dbdd5be5c82303d599af13fd491b7bf9b07db3e6aea033f8b280940b7c8913fba457fbf8fe7d53f9ae62bc0e1d4c4bf81ac82bfa3fb17bf58443940aba0c4bee27fcb3f63ad2cbfe30ce03ea788053f85a18e3e8ca9ecbe7072853f25194a3f2f7d4e3f93dcb9be96eb383edf67b93ebd0a9fbdb3fababfdb5518bfa8f7dbbd88c2343fb48c1c40b8162e3fd79b853fe09db1b9cfe88abf68e101bf0989bebe9004133ff25037bf45fc29bf352a48bf87325fbe165017bf5ede19bca748963f9fb5a6bf0f17d93e155a35bebea049bfd93f87bbe246e1bf864eb63ee080b5be7ec9b33f5c95a9bf772383bd3b9ebe3ecca7533f098944bcd8f1463ee2f60cbf30ec2c3fd0362e3faee100be88fb8cbfd3b2203f400622bf18648dbf01a86c3f7e900ebffd63cc3d2242c43fc4efb03f183475bdbc9c4c3e1fd9cdbe352ebc3ff18f96bffc142a40b4b4613ff60a723f38de87bf84e82e3f69d606c014c5d53e83fd3c3e350d3e3f13e6a2bf0b78a3bf88dedc3e03b900bf0fcfc23eb8a2ae3e20676bbf3a9b61bf8d061b3e0a40173fd5bccf3ed9fc9bbe876bddbfe21376be69940540abb8ed3f029aaa3fc6201240d374afbed4d80dbe28856f3d4816e43edb2ae33ff81db53e52bc743e4080363e9457743f7e3a8abf4084a23e66b165be31d267bdedb2e6bf7bbb8dbee2069cbf4650d4bf267642406efac8bfbb46753e64ca003f9b1911bf00cfeebe1b61273ffb44d73f607571bf3732e7bec76d89bf3041a83ff5a6bbbc6235423fa24e383feefeb93e5742463e7484ed3fa6aa243fe38b03c094286cbe73a22cbf13f269bfb25cbfbfefb6eebe3a19bbbf0becd2bed20e023e1786253f7487533e3bf2423fa5a0bfbe1734763fa4ff8abeb3fc05bf0c4a233e8a8c403e475c65bf0a3cdb3edf2eb7bf8f1b993e211f72be749ca4bff67ca5bf9794f13e342d333fcaf53d3e7e9e04407794d23e5793a53f93a2a23ee2d0cb3f4c22a03f0b8305bf9a20693f653655bf2088703ee33e04bf6dce89bfaa4a40be36d611c09ae545bc3c5ce0bf8e029f3e3004e6be2530a7bed068093f4c5da93e8afeb53fcc16a23fee07b63b9be53c3ffbc32ebf61dc693ff8c632bfac79943ebc7b52bfa07ca2beb412243f3c7a983f1a735a3f32c63bbf84ed423f41ecb13deac2253ea3ef78bf212435bf71c1113e199dd7bf5b613d3c151a06bff60c643f5918633eba31f1be0603a0bee75a01bf1bc0843f952d553fd655313f4843e1bf66e4af3fa3caa23e7bcc1340966ac93e3d4d6e3f2230db3fada3853e1725aebf4706dcbfc3d3f9be5bc213bf7be32c3f88073c3f0887253f089c813f2fc9f93ed896c83e42a2b6bf7a60443da0ff483e3a0fac3ffda328bf935914bf9edfd73f080ecdbe15b92dbe6972c4beb7f2503faef59dbf6546bb3f5a32a0bf22f5b0bf15f9cf3f89807dbf51412bbf85bae93fb6e35ebefd6d6e3ffd18263d57b239bf627c5f3c14bd243fc21b65bece918c3ee878093ff34e8a3fc80e763e93ea0f40740eb43f3b0782bffb18c9be2fe276bfcbe9b8bf92d187bf8139923ef01546be2104b0bee9ae183e75bbd9bfbcef2c3e20fce0bd224d7bbf758a213f1f6d16bfefed01bf708e48bfcfccf13f866439bf5dacbcbfe849f2be216a3ebf88c2883fcb9791bf447180be5882453fb48654bf743e93bd026400c0ab8389bf133b8bbf8bd23e3fedd6ac3e6ee615bf68e719be69094cbe8a13bf3b160d2b3ebfa9313f10b7ff3f7162363f8ffcc73db22ebabdcfaf27bf86f89fbf3618293fbd330cbfd73b3bbee17d88be35a202408449d3be4401fbbf0545503ff8363c3f9338fe3f6bca1c3d84abddbe39c7a7be400f743f4a8b8e3de68e2dbfc6200040bad9e73bf09980bf4dda20bfd1ef283e1137b63f46ee8f3e908b61bdd1c8a53ec7765a3f5bf0923e648ba73fdf2dca3f48e8db3f9e645dbf3fe224bfb005463e7524cabfe895a3bf9b04223ff3e48bbfb43f3f400c230540a2afb9bea0191840921073bf195236c0b219a9be2ba465bfc02d933c7464a63ede0adb3f33e119bf7f9e66bedab23b3ea9ccebbd85aa3ebe75e55bbf15e24bbf3df2fd3ec311923eec0d0c3f45ff1e40014009bfe19950bffa38d7bdfa51163e5ef203bf68a531bfefbcb2bf86db113f2b737abf4490313f9551afbe44bd19bf4a5be1bf983dc7bdf071f8beb8b1aabf62dafebf66ce353e07103a3fc855ae3f73a3423f795b3c3f2c205a3fc2c106bf8456a5bfc7c8a1becfae30be329fe93e4966a9beb3f864be1debd9bf9f38503ffa619ebf873b53bf5012933fc681b43f5ea8433f607f17401c86a7bf40cf2e3f0cce88bfa20583bf3bc8a7bea099c53f93a315400b7e74be5b4b16bfe13ed83f207cb6be798a5f3e9267e53e9789633d279f233f349630bfc86c963ee33bb7bf126fd63f39a07f3ff55d4dbf60db5b3f05f295bf4ca59d3e987f57be8a3f0b3e14e54cbf0fabd1bf351b73bee930b7be7accc93f12388b3f7e6a0340bb20c7be478ba53fd6f74abf3580a33fb86bbd3fbf82cebffdefce3f9dbe1abf8dc7a33d6d02033f0ba8d53f763c32bf9ee050bf1561db3fd1e71d3eeb6f58bf5cc057bfc9e3ff3ee3571e3ff771dcbffef59bbf415e87be2d12b4bfcf89dd3effb1eabf96f5144089392dc050d2db3f2310d73f4fecddbfc84d5e3d33fd62bedec4dabd44d887bfeb97923fd6c473bfd800e6bf0df47fbe01bffc3d8125063fbe7d6bbf8bd0fbbfbeefeabb2841de3eba71bd3fd1f758bf5d4cee3fff6214bfcead6cbf4c79b1bff8398c3ed4d155bfa66e04bec7a64e3fec45ac3f555a383f89f1e73fd68b9d3dae415fbd16b82abd337ad9be146ea0bfd07789bfe724153ee6b10c3f867d6abff6b41ebf6a5c0dbf1354513ff28a6fbf5f2981bf404f0e3f2d54563dea8e983ce06ca0bf2b3d91bfd04d063f96432140c68e0a3e348ae33ea4559a3e328c6a3e5440583f6d1461bf0d4c473f8630963fbdc195bf961d2bbe68b126c0cc2d00bf767f993ec895833ea10b3f3e7767623ecb121fc07d6714bf04b4693e70bf2b3f1310983f010eeebe6583bfbf4f3f733f3e220d3fd2d2a73d809e853f10c6c1be5f2a21bf61e08bbd34289bbfedb3153f7b510c402c37a43fbe3313bfdbec0b3ff5bc793fee3ad13dbcdcc73f010e0a3f65739a3fb1b5833e175ab9bc493e913f16683d3e23dd203f119ad93e8debfd3e319527bf39a8063f635b43bfba62bc3e5b760ebf45d9c0be2bdc1c3e895b953f8ee2573f25a25bbd798c313f7b5954bf1bdd91bf6ede833f5ad78bbe96d516bf90ccf73f02ea453f0158eebf3e4a99bf79b31f3efb7ab23ff60037bfc37af13df540d33e7aad303efae2623fac5962bf2db7a33f31199fbf4d0e273fd06fb23fa8851740d83aa73fbd2fb8be2e09b4bf847d8dbe867ac93f3dfb8c3ff40de53fc204abbf77d906c02926263f1dbee9bf6c471b3f6876b2bee0baf93e9fe992be3ce88d3bd92e1a3e5fe2aebe49cabf3eab4cc8bd20f9debd0114063fa81ddf3e577e28bf16401340b3a52f3fd912153f08201c3efa4b76bf80f278bfbe4b043f045697bfccf58c3e1aef0340792466bee0c676bfb786ce3faf2b903f8c1cb2bdcac3fcbe835606bf5b55633d45571d3f2a18bdbc3eee94beede6e6be66007fbf753139bf2f8324be40b901bf06b1c63f464bb53ff7304ebebaf9ed3f1d40073f147b01bfa0905cbf30d39fbfa063003ef4fc863e4e33a33f7125913f4b2f8dbe6f07183f2184bbbd22502e3e9745bc3fbc50fabfb06211be0c5f1bc0698e0f3f90310b3d7995bdbf9f19d6be2d5c3fbf4510b9be2ecaa03f034989bfbf3db83da696b23ffd68023f08fd0cbfcebc86be853aac3e4aa292be730db7bfd498b6bdeed9073f23f2e1bd773fae3ed92bc2be2ff9053fd4b9d5bf846815c0c246173fc9a2953f5597263e23ffa33f68592a3ed3e7a23ee8bf9e3f2948673f3fcbaa3fd2c5723ff556093ff63266befa889c3f977bdd3e6ba221bf5a44b43b9799afbf8c3255bf01524fbfa6f11bbf0196833e6ccf1f3fdb6ab53efd4977bf3a087bbc2036e3beab78983f675c72bfa0d9173f6ea12f3e704010bfc63c2dbfcc6c813f9f5d064047fc83bfb0d46b3fc37bb93fd8d759bff7fe923fcffbd43f9fdc53be2c69933f755ddc3f23cc75bfe73274bfd682043f19b689bf60931fbfc40585beb0eb6abe8e2ae5bf14c108bd925b76beec8360bf5052893f641a3dbff24e043e8777993f6afd803f96e084be6ea3b23f282a123f8742fe3ee29eb8bd68f7483f782fe1bfe05d2dbf42c3303ffbba84be90c0123f360f133f820cc13dc0e912c078bc333e5a46b03f17666dbfcab9e9be56b2483f46bb4bbf51c81740cb5ab1bf77f59d3ff82f7fbd1710373fe8d2303d18fb0d3faa9c4bbe9339e13e2cccacbe94a8f8bf568671bffba4c73eadd0213ed69ae83e9a6d1dc07da48b3f59e464400ccb2640dedac73e8dc6044053e8f8be8f2ec1bd57e78c3e156c943fb6c1e7bf24de63bf6427813f59fe21bf4a9888bf9ccb04bfc4992840fc61973f4ab4eebeee09953e61aa903fa33aa8bf288ac93e8ff1613f5a159f3e695ee13e572b4a3e663ca6bf32c8ed3f8419773f5b6a113fc885ccbf621ae73f2bb81dbf84ea453e47fb813e9439f13e1d98ac3e866867bf6bfd9f3e2aa9bb3fc4015cbfd39106be729d71bdd8c15ebf90d1943d3a25223de665c2be3d26f2bd3f0b2440061b033fabe4b9bf73298b3f4cd8ba3da49d3abf1d4547bf6b071abf4d4de03df6f4913e944cd43ff4b20d3f3318b0bf6bb37e3e96ee70bf8a1f3a3fbbb671bf918c78bfff9db5be7a67863d6b2653bf7c42f53fb45031bee8a2583f852a583ecaba0bbf78be2fbfa5337cbfb19408c05937cebea288923fc71275bfdda3e5bf5805d9bd5af4853eba02d4bffabfd73be70de2be8d32043fc9f706c07092a63e0aa3adbf5b0251bfa56289bfff008bbfa932c6be1b54283f0307813ff3942abf4c429e3f78c0893d6d682abf03dc81bfc7908c3e404149bf26f6a9bf04520e3f407d05bf6155c3be6309dfbfef65e7bf3a18b53fb96a953f65abf6beedec4bbd495c9cbf2e332b3ebb37323fa48a32bd672c43bde2ee9d3e409229bb0ec40a3ee78630bf5da6443f800ff13ea55b53bf61b73abfc37fd83fa7584abf3d11b63f10f0ea3e6d2fcbbd72a2823f34b2ed3e3fdcddbf44f52cbf804bd13ee6cea5be0ca7f13f44d889bf3d18d43e47ff44c07fd09dbf1a9f0640493e86bfe95c6c3fdd16f1be497904404de03a3f6bef10bf01530bbf7ad7913eb13a8ebf1dcec7bfc875c83d2f13ffbe3ba291bfb6c9823f221240bf0159b9bf760706bf471d2e3c19118dbf150d183f2ed68fbe90f03a3f6db12dbf1827babc976b25bf252479bfdc528f3f232cc8bfb6df6f3fc27191beb8fc05c016b11d3f79c1e03fe222ec3fec3d13bfbe7dd53dec4d693fd9a226beefe1bb3f199e143d3196563f8c2cb13fae29073fe13eb1bf954096bee3d2643f9355673e69c2124058a839be40c6193efb68a73fd40a0abf39b4dc3ec6f5913e8bcfd9befb73403f74ec0e3d245cc3be40f01b3fd3639b3fb1f7d1bdf995c63f360fcc3f151cba3ec549c23f4f11a6bdde4f903dbbdc0e3fddbf383f4658873e88307abf3724aebffd34b4bf68accd3eca61f3bb7d0b003f376060be09036b3fb3bdbe3f67dfd9bf0e3e9b3e867a8d3e6c031e40174818bf3506883fa0fb2fbf38f94b3fc285a03f80ccf83e7497a6bfb5146b3fcddca83d6ac621bf37b41e3fd098953f6914a13f248fad3f2127aabe284b4dbfb3066b3f34aac0be8232fdbf954455bfd988b83f65034e3f29f5c43fa559833f4cc53c3f9062843ff3a22d3f5f0c9bbe85fc51bfc5d6873eb8be17bf8873c83f1c7e14c0425fce3ef098063fca1b983f3ddc94bf8041343fa8772cc02951a8bf7eed20bf9472febf6bb0ab3fb902ec3ea27eba3f98b46dbeb56234bf9e7fe13c80b7d4bf8475363f6ece96beb0f9363f103bd4bfd5bfa1bf6a5448bf7e4f8abe9c9e973f613f2fbfdbd44f3f023cc43ebf39563f189525bd372cc63d7ba69bbfb4c91fbd441e3340c2c4883eca929d3f1a432c3f1ac021c0bdf48d3e88695bbe26ad5cbe44afff3f848bbdbfe8f590bfd07619bfa465cebe13a510be59f0dfbf5d8616bf0216a7bf1c89b2bec7f2b6bd108a45bd3e492d3e5a67f1bd4fce953d2485fd3e7d9741bf85e2f83e9418823fd9e0133d2891313e3df8923eabc7d4bea0468f3e77b9bc3f5619a23ee8b8123fc8fc5b3f1003d83e9e92423e732f9c3f9ae965bed580653f4f990dc05ba110c015b3debc76e46c3f9752043f46371b3f409363bf0e552a4004f9a53e007843be0c5ebd3ec625103f98c54ebf2c6ab9bb74a11340220b9d3ececf1dbf4b628dbda2333fbf47dad73e22e0833f157c073f2856253fdc9899be72c80a3f18698fbf1a5ed6bfc357b13e52b426bf15280fbf59331bc06e59a2bf8aa0e4beed547e3ffdb63e3ff2a7dfbe8391993e218fbebf47da0dc03d18cf3f95ee863fe5a7443f7824b5bf66c5353fe62d28be57041f3f6f7990bf82500abeedd186bf448b0f3e6e7cee3f9852b13fabfed9bf0606c13e41343e3f88aabb3ee1d121be6ecaf93ea8d7a43f49f6dd3f45723fbf9818983e5380d33c8435b0bfebe6653f4ed0a03fd450fa3d678e7c3ff75c09c041b38abf1d4e24bf9eb5793fb243203fe9b69bbf7dbac1bb8ad6323e0ee2603fa42819c01c9996bfc9e474beea59b33f8084c03ed5e0513ec30c5c3fe9438b3e3e49833f045f063fd54a8c3f7fd8e3bfb279363e1bb9c3bfa5b3bb3fdbb0863ff35f6fbfe2ec87befa7202c0ca22323f9e3f19bf55bf4e3d355f3c3fc1050abe76eadfbe59450bc01085ab3df6f8dabf703b083e45c672bf284ae83e7f0a0a401048e8bdc1e5d0be621f27bf7531883fe5d48fbee2e9bdbea38d473f46c0ec3ef2c7633f7dc7b9be3e133bbfbf0b5c3fbddd9cbf9c333f3f7490f2bf7201c2be5495b9bfe8749fbfd95a393e4179933f28a4353f9a420fc06529643f250b583fda0629bf8ad0d1be5a6f1fbf6becbbbdc5305ebf52e0763efc90cbbfd1dcd63f95c68bbf2c0b06bf85c5123fc8654fbf0112acbe37682ebe112732bfe7065bbf011259bffe944a3d65febdbe1bd47e3f8791b4be85c7b9bf91291fbfbbed803f5945c6be33ba773efa9c69bf431f9bbf61077abf9e238b3f27f5adbf0d38f5bf6a8224bf03f5993fa23337bf726ca0be01d5c73f353773bf7057243de949893f7231d43fd80c6bbf1068c83f356895be569af7bf7b128f3fc9f20dbfa21090be9b2af4bf1121ab3ff7b914c0379f05c05f17d33edb5e55bea8adc0bfc477983ffb2b483ebee2ed3fd82e43bf10e19bbee4cdbcbfe0fc4cc04d020cbfabb71ac00180723f13c64dbf6201da3dcb41c43ecf0d43bfb5d8513f5dcfa93fa78eb13da902963f9ca0373eba80853fa8f701bf799fd93f834dc63c879d2f3f63de20403b70983fa30d243fe4164a3e4f55f03e033bd23d80f5ee3fb4ce563f1f0a963e9e0819bf2537243f4ee1063f504807be0cc43e3e1af4023ef267c2be1d869c3f76feb6bdc1a6d33e8e213fbfa034efbff3767f3f37aa80bfbf5fafbfe33aa03f5bde83be3c4aaa3fa9a7bcbf5be63340d9b3fd3f036b833fcac0a7bf0b70ecbe28aad6be526970bf8e23ffbdc5e138bfb25a8d3faa92ba3e833c6fbfb21b9a3ff5753bbf0932d5bfec742bbd8ab0b7be13ef593f3230943e8c918bbe2b3d30bf72fba4bf2d704c3f57c8cb3c4644d8be848afabf7d04093f3bb532bff242b63fee7b21bfb27eb6be80cbde3e35c49c3d2d1d0cbe4a5a82bfc097ddbf2bae053fd03e05bfdf7201bf4a15433ee10a973ff970113ffe2f05bf5cdb2a3fb9eeeb3da0793fbee8f99dbeebb2063fe71479bf690cc7bf228423bece85aabf39a050be297ca03fc6c469befe561ebfba8619bff13b7cbe14971840e7ffbf3eef0fc7beaea71d40663db93efde60040e96873beeba4d1bfd04a87bf11246b3f21c49dbf60eaeb3f52bb8cbe142ff9beb3521e3f78ae64bed0a5ab3fecc9b1bf9d0e0dbe244306bff9898cbfe5dd56bf6f1305bf97dde53f996f2a3f9d26473f402f3db9484f55bff46ebabf06a8a5bf29ae74bfae793abecf777c3e4a93a9bf6854ee3ee0666d3fda6197be137ad1be6220363f5d191f3f0cceaa3d6ed5973e09e048bfa9b0d03f619f0bbf913011c0cbc70f400339073f3a44b8bf4a9af13e5ccd7a3e11d8493fd940d8be0be5db3e767df6bd2744543f65841fbf5526043fc66703409674febe4ce0c63f589c4ebf7538ca3e120f8dbfbad4983eb4ab82bfa3ea7e3fadaea9bfbacae03f193fadbe64fc1e3f4aa2ccbf083a33bf2553533fab7a093f8c0a403f9d2c903f2b0ec7be29542fbf5c3de53e8c27193f971ca9bf8dd11c3fe8e86a3d6558e8bf2ea60cc0a070e1be2ac1393fd065a5bf8de952bf066f84bf0f1c21c0618265bf62b763bfc2f9aebe5ccd473f1f3ba13d1f862140201c0b40d745643f4edf383fa921473f99c72cbfedcb37bfe187fb3e77e1a63e34a49c3f2d55423c0696a63f2bad99be8802d73d87943d3f9dcbac3e15781a3f8029433fee0fe2bf944cb83f92e4a2bdb41e6f3eb1d81abfc91fa7bec960d0bf1c5d5c3f2105073fb33880bf505ec43e042c2bbf89e401c07e8f893cef3a19bf46c3333f52ad1f3f6a4d663ea0311a3f5e7bdf3fea747ebfab93f23e38199f3f6d26fdbedd58df3fc67d1cbe958080bfa52f643f61e8a1bfe11666bfda5c8ebfac80c73f4b9e233eac1084bf16ae74bf38e7bbbe51f6053f7a6cfd3e881e22bf5afc933e24edc83ea49a7a3ed89abcbdbd4ff6be64b3273c43aa60bf141077bfbd97f0bef7e324bf83b8aebf9ca0f3be87d3dc3f786bce3d334457be219fb7bf36dc313fdb17bb3ff6d1a0bdd7a103bfddb79a3ea4544cbff084503f0e44b7bf344ea53de139a93edd1302be1970903f9a037ebf44300a3f4bfd513fe15949bfec0c0b3f6cafc3bfca68a0be855c183d925b053d04cb913e73f9bebf6f0aca3ff7becf3e93f5913f8d6f88be1d086ebf5c0a11be3c70843f96463abfc010013f79b9893de87fed3f2d57883f5945cabfcaf7c13fea377b3feefab4be94cc91bfe26317bfbd521bbec8ea793d1f8f973f34ca77bf925b1c3fecbfc8be1cc9ee3e665ed5bc0e9653bf97e1313dc02fd3bae0a42e3f1308f8bd457587bf6073933ffe58713e5af3c7be96e892bf8efd0a405561493fa6c4633fc5822ebf059854bf5dc491be01c8cdbfa75cfabece8a3e3dc52b82bc698abc3f01bd94bf411bd53f9015293ffb6dd9be805710bfa031dd3d1273acbec716533f246f9b3f936615bf7e82193ed6a46cbff9db9ebe7810343ea18fd03e95fbc33f285e0b3f08a6a93fe7aeb6bf90767abf87608cbe724cf53e5cbc16bf002fd63ea9ae76bf3cc5c83f2156a2be2261303fe49c88bf9e9b533eb2636e3f3df403bfe454993ff84bf3bf5257323ff43258c057bc263f5d9fb2bd2cde9e3e155abdbf8793e5be41c6a3bf3e153cbef30d8fbf5f3ebbbec0449a3ff5b05cbf6e05283ffda285bfa4de2c3fce8082be0911aebf5d040340a2d9ff3ec85353be78c4c53e4115dc3e111d8ebfff859f3eacd14bbe3d55b3bf94f5273f3f66523f6438ad3fee30c43daaef55be1ae856bf9e5f15be27299d3d6b7139bf620c8d3f0e219d3f49d49b3f0d6e513fb1d77dbe20d446be5f10b23fd17b663fa8d438bfe8409a3e7b858fbf8a92603f5174d53eced93d3f8254a83d02f17ebf4ff45a3e665fcd3ec5b315c0f31438bf34fb4e3f74d893be99a5e53f3531873f613524bf44a56abf36be053e81d7183f3ee8a53fd0e9a83f8872e5bc357d593f3f04143e9b41943fff25a73f0bc5593f81b30ac00fc5a1bfc8b0053fb0c62abf3378d93f543818c0acb424bf550c18bf0ac8dbbf4841c03fbddaf73ff2d6763ff3569abe10ac09c0ec84bebc61ed3dbfd6e9aabf980ce73eb65139bec539bbbfaf1c1a406dc45f3e9f481e3fedabafbd5cd408c0cfdf67bf34f6663f17b6e53d6e824b3e37bab43dbdb90e40255e403f0157a03f9c34fb3e944ed33f1ebb11bd900dc93ecca5183e33f99b3e532d793f1fba15bd12bec4bf0a90a9bfbea258bf481e553e95ef46bec955173f7bd9d4bf393b8fbecd6ad33efd141bbf66f362bea8c312c03f7d3c3f9811febfac4384bf92ef45bf1b6b683e04a6703f04f1223f7a169bbe32622ebfefa52fbf572dce3f559626bf96d01c3fa95a0bbee419a2be5736e13e51bde2bec471a2bd1d39063ddbaedebeafa6383fb67839bf491e22bfd86927bebc88dbbd64570fbf26ecdc3ecea6313ff78d66becdcd333f233aa0bec2eba63e24f83cbf604880bf0029e6bfdb0337bf4563ed3fcdc382bde3399b3eaffb0f3fa9a566c0c799973ec6b0dcbe1890103f6e1ea2bf900e57bd6e28dfbf8ddd95bf216ea63f89a68c3fb3feadbfca234ebfa882c8bff4118b3ff54382bfacd089bd265ad43fa2640fbf15aa73bf8b43cebf2d72463e3204173fbbf3173fbfe0bebfc65f103eb2fab23fe3cac93f0454e33f2c5c773f915e043feec79abf3ea19c3e91f9d63f6179cdbda29f4e3efbbb41bf91f69cbf8724323f3d1ec8bfe0cda1bf634d533fadabe2be9506f73e8346003f1712c0bffa241a3f0544b4bf4616573ee9388fbfd5a090bf5b055f3f605ffe3f4b73bcbf2ecc283d6b02c13fe0436e3f64c9243fb06ef8bd7f60453f0df3a6be43d2bd3f2d172d3ebb2721bf68e3c33f261ad23e9a14823d543285bfcb2a4dbfb90413bf75e4ff3e7bd9a3bfb18058be38c424be21324a3fa628fe3f9f3b3ebfeeb4963e976320be555301bf3700e73f8e228a3f0aec44bd610de7bf930525bf9d9daabf37d2d93fab481cbc3d80b7bfca8299be88da0fc055f951bdb6d9d43e7e2e25bfe4c880be5830d13f764298bf9ab72fbfb645923e6e32a33f7780963f21b29dbf8d0570befe3fdabfc8da863f168a213fcaca8e3d98e255bf86054c3f8c20b7be8bce7c3f25b478bfb6b55cbfecb5353de336663f3a0741bef8b156be6ada8abf67d296bff257dabfeb1b3dbf50e1adbc27bda5be3e90e33d307028bf1d1222bfeb6193bf35dd55bf01ba5fbfa86cb33e17e7c7be8835cf3d9bdb803db1829e3f64ba7f3ffce4ff3f1b3b58bf8bc9a93f54db883f286374bf9b91a6be5b8cfabe0ce610407622783faa3142be7f4fb23e64ef8d3f8316d33e8da52f3e3219873fde7d81bf9c38703eedfa5bbf718e0e40ea62a6bf002dc3bf8cafe1bd7266093fbd3982bfd16fd33e0d8430bffd1d3dbe4fc0a23fe4f009be1e28773fdf86233fe0b544c02b3f36bf77fa363f4cd7a33f1b16803f6461bebf6c84093fc1fcb0bfce33a73e06d7bdbe2b51063c1101e43efd958c3d83b97bbdcf918dbf77aba53ef3ac9fbd09cd0bbf3dab653e5b8837bf068a22c0d25884bf56301bbf6cf53d3e94462ebf2ba0143f3fc5ad3fc9a4063f26b2bdbea5a8aabfe7f4e6bfc6d299be0405b5bffe5038bf67825e3e02d7b33e1640e43f4921fb3d219fa0bd9b9ee3bec5ee9abf7d2714bfa08787bf49eddbbf4409a93f135c8a3fbfb6a53fda7e6fbf3fb10dbf6e59883f5a62fcbf7d144e3fa871793d1ceb173fa6251bbfe2d2533f0e86b93fac3e22bf82324e3ffcc60ebfbf44fb3f9ea3d0bf49e2bc3fcd28073fb0a5803f496d23be7c6f32bfe28ebcbf8d8e113e58e0f73ddf19afbd797012bf11f2dd3ed32dd2bf49b49ebfa4ad8f3fa82fc9bf34ce76bffb4e2bbf5907fbbfd0a304bf530ff4be11598abfbe7f8f3f57fe86bf5ce75b3f0012ce3fb1d2cd3fce12253db8c65d3f7475d0bf25856ebf3583a53f3ac602bf87aca53f190f0140c73f17be47bee13ef270233fce3f2ebfeb4d793e5087fc3e62e589bf47849d3f415e893eac028dbf2d94973ecbd5153f88941d40a431043fa385c93e741f47bf91cccebdf02037be67ed883d9f600fc0b39f6cbf2de542bffc2dc33f3e684abf672ebbbe76d8e3bd97bf9abf0144bfbf61a6a8bf00cfcdbf6f41f2bfd2b880bd2de443bd105ae2bf74f4733f22a8483fae3a89bf2ea99e3f42e665becca7843e34cb2dbf633d873df2330d40e487953f1b74adbf2ca1b73f7b04b7bf7ea9643f061832bef464943e3030233fdd24e33e818099bfd489443f17132b3fa9e25f3f4280003f7d3ca63f2488cd3da41bdcbd4ca1c33dd9ba1140afae29c062ea1fbe5c66623ff2c9843f187594bfdc480940d77af13f02be313f3690463c892c15bebebcc1be50e6aa3e859d8d3e1949b2bfb3bdd7be8b993c3f7e61d2bdebcb5b3f892b08407fae4dbf19d7a7bfd07cd9bfe87acebfaf8ec13ea12632bea630003f091b723fc1eea0bfc6375a3ea612d63fbb6ed3bf4461adbf9108c8bf39a71fbedb6010c078cd2b3f7b3430bd00bc4c3efe151d405f3086bfaf84adbf62f010bec0b66ebe0cbdbc3fd4ebb53e8e4ade3f8fda6c3f9a8409c0c236dcbf3206cfbc1a88ad3e90f3483fff0c14bf31fb92bf2932f7be9cc55ebf2b1f0d3f4576863facbb35bf2b960cbf57bc04408bb091bf8523743e4961c53f03d8d03f7e9dcfbed204023fe892663e74f57cbf57b1fcbd1b23773e4f6b53be2a2b14bfbcc1ed3b83e3b03f9f06e6bf2c67a9bf22a1a33ea075e73fd265623efef8f2be42b67b3ee918213f1d48c83fe6260a3e64a8e33fa057513ee9d0a23e865a0ac06823833f8206bc3fa63a00bfa8b81dc00a845fbf5c9db8bfeb49eb3f6a493e3ecef4923f471e823f42def2be6bce4c3f8bbed23f104a3a3f6c76993fa32f6cbef264bbbfa325d4be0abd193fceb72d3ff669ea3fc0be293fd58ea8bd6be4a5bd57ac053ff93d4abfd057c53fb7972c4099f6f23f0ef3183f9fd2a53d1c39703f03aeb33ee4932e3ef9fde13e483be5bd5289983e64c4bdbf610fadbf60f9f2bf882854c0a8d2da3e66ca03bfc3403d3f28d4323eddf84dbf12000f3e47f497bef3ebad3f63dc10bf4fece33fba9f8b3f11ef48bedf5abe3ec31e2f3fdeb308bff8303f3f4695e4be22ea25bfed37d23fbfb4d7bffc54ba3efffc153faf84b23e37382240f08410402673f1be9c40943fa26f06bf48e59c3f6729253e5e4d30be6f29debf8d5f3e3f6dec9abf4610d43ef56278bf783b6bbe9a1c4abd710c773faebb8f3f9394d33e5dc69a3edc4ab13f1f97253ef2c018c005b418bf8354bdbfb87ab03e0e3dfa3d96ba53beff85ab3f52f9c8be5ce0813f51ec04c0a07093bedbf8773ec5fe12c0eb982c3fd8f6e53e67a90f3f99bf313edb7b823f6bf056bf3493253e1b0788bfc65431401f10ec3f9831ac3f5655c3be46f9883f683164bf4a64f5bec5807a3f5f0b64beee63e7beeb1f193ec8707f3ff4b8ccbf88d9f8bead6c87bfd3b4133f15d490bdeee8a03cf7212640597d13bfbe99bf3fdfb8ef3e5a363ebe38ec6bbf23a487bfcc47073e1ad2153fe2bda0be82f19dbfee89afbe28b6ab3fceb7ae3dd79dc3bf3d7296bf23868fbffb6578bf2af607be2d84afbedd6dc7be377f7f3ec4b81cc0b2a80fbfbea93dbfa8b3923c5acb1bbfb793ccbf512cb3bf736e94bf5340183f6db9eabe8b990bbe4dc639be1e9554bf5f7ae5bf9d27753d268f403f4a6acdbe9069bdbec56e36bf30179bbcd8239abf4479c6bdc29ed03e1d50ac3f0110cabff46991bd4cc85e3e5bb5763f77bc9abf6acf4dbe71fa1a3f58f82cbf1562cd3f44f1a83eeed92d3fc9838b3fc0220bbf135f203e6d107b3fa6871cbff9b0bfbee33b0cbfe078aebee7d0fabed4ae4e3f042da33e890b254028e002bf848a3e3f574f0dbecb4b70bfff61c03f78a664be671678bfbb81643f155cd0bfa7a05c3f2e51b3bee58e113fcc6d2cbf9a10333f2b80dd3f44debe3fc95b143fd062a93efe46be3d4eb45cbfa56c883dbaadc0bede55f1be65eeff3fdf0b783e5591353d19c9203f723211be70b810c038ee61bfee57ebbfefeede3f968da9be1e76203d96b6553ee020b83fc12d133fdca1183e271e06c098fdf93f4c31333f42ebd0befaea97bea437eb3e03b9803face435bd3cecbbbfa3799ebf5c39a4ba5efd693f9430d53e4f63ab3f6b382a3f64f4f63e7220b1bfe1534fbf8ee1b63d4f70b03f8d6a363ecf5086bf2b92ab3fcd0ba83ede42e9bfe968583f2e2fa0bfccba15bf5e5a91bf86133bbf9ac915bf8f80cbbf50f1e6be7907913e19b4853f23a3c6beaf0f393fabef8c3f3cb660bf50635d3f3cc636bf39929b3f3ab00dbf1ee33cbe2117ba3f860eb9be25318cbf409a843e66f0923e412a0d3fa3a95fbfd2c7843f28336abf43c3ee3f0368c5bf373c963f3ba1473f3e8bb9bf5686eabeda855a3f9f03dd3fcf6ed43fcb50ef3d42a9c2be588cbdbca51fc13f4547923feea7a63e0213dc3e2d326ebe184dafbef49bd23fb6af70bfb875453f290a4abfe4f8d6be2ca0c43f94439e3fbc29ccbf2d037e3e8143c0be76ba7c3e75b8af3e5fdc8fbee32f64bf778f943e3193f93c6a0e493e8c2c3f3f4b553fc0764d51bf87ede1bfd879823b7162b43e96e82ebf9c53f83e752c2b3f336a3740beb807bff0c5a2bf3f5dcb3f13d1f73e41b8b13dec11973fc05a493dc0fcdebeb1c6d83f39f8ae3ee1b1a53fdbd9363f5969f3be9c5defbf7e4ef33e52b233bfc49d163f4c502c3fcd48d73f84522740b345ef3ec38f14bfcd9fcabd27bc953f6b429d3f2108eebe947e673d97998a3e68264dbf5dc6c2be4cd3c33eb46cd5bb5b0af6be01b4c0bd38f9f13e15b2713e1131573f253b49bf46f3df3ffd319dbea3fd3abfd6f091bde97a18bf0464aa3f50cb63bdac982fbf8df219bf49b8e23fef98033ef303133eab8c83beffb88b3f645d99bfc25658bff5c7b63e96e566bff40d4bbfc36d7d3f773e1bbf289c073f8916a9bea38a0c3fa7302e3e6e7919bf185733bc4fc499bfa792a0bf3acb383d0fa25abd7b5ffd3edb26573f13e3a43faeeb1d3ed82b4fbd2da4c1bf61de3b3fafb413bf05ecf03eaa2b083f17e19d3e28dcae3f10ef8a3d0195a23e5c73623e9b425abeb825423fda133fbcb846093fb6917c3edef577bfc803923f1b13183f123223be65b0b2bf30b206bf9b2627bfa3f0f2bf8e07b9bfa6fc2bbfda83b2be0ec663bd5587e8bf309498bf943ef6bc15c1db3e103eab3ee71e16bfce68783f5cf5dd3eaf4e58bf3d4962be4dcc223f666b81bfd591f23e8cf2113eb50e72bf838dacbf1e7b5fbcdc16d5bf89ff39bf45ae0a3f80ceefbe936d2ebf33515bbff699edbe09a486bd8e64ad3d824d103f8975e73f82d04e3f931d3abfaf457240c775853f54688dbf0dcee8bf286393bf843cf33dcef3cabfd93e92bf66bfb83de8c27abfeffe23bfc24f223f6436893fdef3bcbf24c4bbbe44c4fdbfd81f363df11005408da495be78e79dbcc3793e3f1cf0d83f093df33fcccb3f3e2d47e53c8fa0063fc1bf93bf50c2cc3f9cb10e40a69828bfa3c3b5be9d578cbea36f06bfcb0b85be034d0cc0d8ffc23eec8b17bd7e41ec3f"), dtype=np.float32).reshape([1, 3, 32, 32])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
