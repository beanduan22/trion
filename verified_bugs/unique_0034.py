#!/usr/bin/env python3
"""
Bug #0034: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['broadcast', 'cumsum_last_axis'], ['normalization', 'softmax_axis0_residual'], ['fusion', 'conv_k7_stride2'], ['branch', 'inception_v3_branch'], ['normalization', 'batchnorm_relu6'], ['layout', 'conv_transpose_channel_reorder']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0034.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0034.onnx")
INPUT = np.frombuffer(bytes.fromhex("f160c6bf5466f03f5de4c8bec9f3503fd70ccabfce9f8e3f711e0cc06f9f8bc098a254bfc568a83f62bfc0be70791c3f9726ea3f36811b40afbbd93f269fd9be6ec16e4081124cc07cde3e40d4357d40d6c5cb3e5372ac3fda0d01c09a416a4016e4953f3ef254bf87e1a23ef64cc2bfb4ca8cbe01c42ebf68e386bfe865e33f7808373dd1f97dbea744a33ff59c7ebfeae85a3fbd0eb1bf4f264cbf82f686bf49c5863f1a1b8c3e88cbe13c8511214093fab4bf4fd0d23fa78c82bfc834213f791e223ef8807abf701da0bf430746bebacf05bf4e2907c076d729408370fb3d88352a404fe33640df8bd3bd4c2e174040c2573ed56a4e40878d0240960c7e3f66188940f1e0a53f26e4eabf4c313d3f0280253ee490723f4c8143bf17a675c0669b71be92fe9f3c3d7becbf2953ba3f1150aa3fdc5340c046e00ec0934c9e3f66c5b8bd31903640d35029be5b3581c0ebca194059c3523e312309c08e350c3f1769543f486c993f9f3da9bf2d47f73ecb9bf33fcbaf22be210c52bf20da0d3f1e2514407e4996401836fd3f32acae3e053c373fbc55a8404e37acbf1add44be2cfdcbbf30ad93bf1366e43f9a37a13f228b7e3f0cb11bbf272fb4be6a11ddbe30c887c09090053ed38d45bfed83cebffe3451bf6724a03fbc554abfdebe143f3e2c18c08740bebe2387a4be4e3d0b40a67ec13f42c480400bd1b13f65ebc33e8af3f1bf83f783c00e132bc047b6d63fc0eff1bfee3e0a3fe07b803e378c7e3ec6ede0bf732c71bfaebccfbfef7aa840b1ae243eec99bebe8cb3de3fcbeb923f4daf49bf0b7c5c3fb769ef3dad3f52beaf081c407b3f4ebee75570bf214cb8bc5f184f40a975504098a25f3ffb016bbf723459c08980353f0f7afa3e8dfc7ebf3f5371405cd580bfc1e2f0bf13822940f8ed113fd4f1113ece0480bfecaa1fc0e28bb23fb5aba7bf7f02e63f0e2178bfb240a0be696084bf1f720dbe3609ed3fa76fb9bf0ecfd83fc0bf10c0fd0016bf7769c7bfaa3809bf976532c0231d89bf1f3ee6bb13fe294057f87a408fd695bf2090eb3f3d38b3bf8b1f8440cdd4513e46999dbf41a40dbf6c2cf8bee81ebe3f9ac6b9bf621fd9bead942e40342e2b3f67a56bbf545589bfe744e83f29c2b73e60ae5b405120014054daacbf5669aebeaf2837c04abe2bc0411a3a4042338d3f9d0d8ebfbb2c45c03c53edbf95875140bf87f53f4f1248406116da3f5e771c400ee08c3f56f3723e031552bffa6515402d35fdbffcd8563e9a27d6bdf8d3993ff33aba3f223e8c3f70e61b40815c1fc085b1843f3b9a253ed329903ed10ed33ffe95c73f162b923fa1d495bf82f9263e27c334c05c12113f4adb13c0602d823fc2e7a23f186a8fc0243e2c4008e603c015d3803f11b8643f4017b33f5b76473f33c030c0c20a8c3ee754a1bf10265cc06a2bddbe3b9dd4bf8b6d813eaea4fdbc901d47400415994033d57d40d3a36f40feb323c05d5d8bbf2256a6bfdfea19bfe0f43fc0cfe1893f4a1ccb3e882c6b3f22259e3ff6ec0040f6ff7ec0e3c9a73e2c9648bfc4bf8bbfb99142bfe3c35e40c3c3433f28140b40f1d9f7bb2aaa5b40378784bf193f383f18fab53f56c645bf1dc1843f77a5fa3fd6ca0140036fb0be3bedbcbd04cd1640c5890ac09d00d0bf5af34c3fec408f3fd67249c0e2eeffbfbc51354029237c40644b64bf42a7fcbf819a4e3ff4a8a7bf3222503f6e7e1c3f210ee1bc4f53d6bfac5c78bf848f92bfe059613fa138873e6bbf20c053f6763e400188c077d0a23fa74c1bbf0779aebe6fdbab3f62af3cbfe7f02c3ffe54eb3f27ff5440d3259c3fbf001c3f13dd22bfa12cf8bc7966eb3e774e2c4060fb8b3f0d665840c48aa34044d60640b16b17c04fa9b23d7bfd02c0eb7bdbbd8c2f623fc451a7bf838d4e403c0fe63f1798d1bf77b8363f3c4dc6bf890c2f40725ff8be95e614406e0e2c3f28a9453fe9be0e3e9254ccbf620937c0ad53184046fb60bee2c6904048c8fc3f7c068240dfd1b2bf5663103f9546d4c0d62f60bf97826bc00e3d3e40e6c3aa3d7d5f21c09edcc3becf625abfc0b38f3ff8940b404674cf3fde6e60bfa857a6be678828c07e6031405036d43fbc339dc0bf73cd3fa237d73fe87d8e3f1efed53f92b1293ff492883f2025853fbfa34c40e7bf2ec0ca0f6f3fdbab1dbff92db1bfb9e783c03a682940284f96c0b069b33d282e953f7bdf37409cbe83407a4c95bee67bacbffcf508c0301628c0692facbf5f8727bf9c2f6b3e935692c0ac48da3f5fff71c0413ccabf9663264044b426c0734d8cc0d6b900c049e8aabecef1b1c0327638407fb63b4077e2e4bfee9f4ac0808cce3f9e211bbfd118edbefd643e3f06f101409fc9933ea360523f7ee110c09b1a1abfb73df33ff1e4883fb9250bc09a5b2d3f6652923f568c8b3f49d3763f08cc0a40426455409c0d30c0a7be8c40665a11c03f4dc5bff3db513e57674bbeaa7b43bf97408c3e076c2cc05bf5bb3e3feacc3f5be7f33fd5abb03f3c44a8bf7fbdcc3fc305203fede710c078599bc0ea680fbfc0c4184091508d3f35a7973f6b3fcebe14c70d4010187b40c45311405dba38bf4d9b8040cd5c69be76a8bc3fe3ad6d3f956b653f7d618fc0141f89bede4fc33ddfb2de3fae30223f0fbd8440c51f8a3f79f773beb9a20bc0ba2d6d3c7f7c6f40d7adfd3c529f15401daeb63fecc21e40e7ffa2bfb53a7dc07c74074081ad86be378f16c04759de3f482b3b402535413fcc1a13401974223fbfa5a6bfc030a23f733ca8bf65e602bf11a40dbff81be0bf999f883f2578113ed70a6dc04e19d0bf30f0e8bf026bbf3f6a1992401937ddbf04841fc061c42dc032ea5bbefdf38140818c0e4043b681bfee4269c07cfdb6bf769e4f3fbbbad0bf40dcd6be7a9b993f714cbcc05536683e52a342bdcb660cc0439b20bfd42c463e48838c3f77e3bebfddd865bf1d60463e2ffce7bff3c4113b4dc7b03f63146e3f999c1e3ff1c7213e008745c03dce403e2eeed23f2134d2bfaa51273f01aa96bf5920bbbe85b575bf227e273ff4cf3a402febdc3e5b929540a921b43eecf3a53f2dbfdc3f37140cbda43f223fcfdc8440e11206405e83aa3f91ab0bc038f70f3f24b7733f16558bbf71415bbfe1bc05bf61d633bf328456bfabaa2b3f0ed7ccbfd98585bf7f223ebf46598b3f800a0b409a90ee3f008dd63f81269ebf5cff6cbf0e1cbc3f3f3f1cbecdfec4be98f8aac0de674a3fc492b13f27783840cfd07d3f419f2cc08605e93fe82b8ac0781c62bf56ed00be49f2493efb6a3c403a17464016b67ebe734287bf4825afbf97f5e2bb3790833e899e20401b9417408d1ed3bf9492edbd570ad2be58b804404aac973fb3c38c4070841e405d650ebdefc706409d1b67404420573fcbdd223f0870f7bdf16a05c01b9a77bf60a369bf1c8f153f858b7f4014ff5ec0f6097fc02e65f53ff81681be24ba9640bd0339be8926ab3fe5197abeedf587c06f6009c0b96905c053a7bb3fe651be3fcfb47d3dd059253f78a3c63fb20d93bf31a6b6bf1570f83f7649debfc22613405ef0aebfc1b047bffba76c3e885667bfaf5c95c0fc6088bfbe8208c07616a4bf003d4b408335094071b8973fd209e23ff53144bfa7910440b6ce29401670893eb28d1d40fce79f40fd8c3ebf4c40334052703d3e9c38aabe873e5540584118c0517b014013d8fcbf23d0ea3f9d84ba3f5ad1bbbfdb4ba4bf3697d63f1390223fc6ee67c092fad4bee04ac93f25e21f409b9b73bd24dad4bf0a6b3b3e6ad728407ec2c7bee0d982be405619c018f8a83f9e01713f9aee1f40c424823f9e189a3e27934440a3a9a2c09ad5e5bee2ed63bdaacfd7bf1dcb56beef92bfbf859e94bf2635d9bf50a91cc0c313583e7455a73f27a888bfa788ddbf1a935f3fce270240af61b0bf8a45983e8df228c0337312400b96d33dd21d4240a8f1913fe1387dbf6a7537bf4caa523f8d5780bf84a64abfcda9cb3ee9a3febfbc2fc2bf02fabebdf00cdebfea78c33edad6004077381740988daebe5aa04ebd1404c5bf1a8597bff504efbe8aa570c0896e1940ff8495bf3ccb9cbfd2fa03405bb3e2bed1a64abf966c083fe25316409f1408c0f35bcb3f83d9afbfbba024c0a8fcc5bfdece853eef0aaebf3827aabf1b2c23408c778e3e736d8e3f91193b3eb1554ebfc7b49cbffc87eb3e100ccc3f6baeb93dc34f1f40594b6ebf8ac26fbf7fbdc7beae5215406752d9bea83c7340906123401277713e1413094062551040fac92cc07d9a19406630cd3f65c614c0ded8eb3f4ef2bf3f657483bf6fac3b40ffb8acbfe0a9613feff83dc0027ce03f3df5a53f27bc93bec7b8c6bf8cd38abf77a312406208f63fae49b5bf1ffe90400454fdbfbdbede3edbcca03fe87f35c0d0db6ebf1d97f33f8662ea3f8e6ba0c0ad9127407a114440931b3fc0f4d97fbeb25cd8bf262175bede3fcabf440f8cbfd65fcb3f6855843fcf3e39bf597197c04b3d0e3f589c673f08d384bed0dda83f1ef1dfbf4d58773f01f98140ce5ae0bf73023cc01a815fbff666893f59fd993f608d98be6f38c03f4b6163c0da0fec3de72fd3bf22c00a404b508bc064b679409a153bbfb7a052c076fc11403f5365bfd40342c08471f63f80f6bfbf266c19c0db18e43fd212fd3ec0e5bd3fbefe7bbfa6d0843e1af62ec074fc7ebf7818ae3df34cac3dac10eabeda38a1bf0bfa144051e0fa3de5a13a409f3befbfe765593fca360ac093d4abbc7df875c0668821bf24774240c51e4d402a07223f7357b83f16c28fbfc6c32d4081a3193fc890b9bef97e11bf64ea38c086e6a8bfce1fb73e951087be33741ac0d801873e50e44fc0b75107bf18d422c06fd6a53f7e51243d8cc4f0bebbcb7bbe486e583ffe359a3f6e3afd3e564b983f75cb0b4025d320c04f0e04c0f1ef0ac04faac83e128237bf41c8f9bfa4302ac031cec23e8ceac7bf4c1d68bf68d189be9b5c85bf8aa75cbfe75910c07a450c3f8e177e3ef801aabf8fe6584057cff43ff37640c0a05e763f54071a3f963eb83f44d9cbbd48c97b3fd03170be215e104000bb6ebfad946a40961893bf24b7ee3fc0ab30c071a10440b034e0bfa42a8d3f308e2c40e9c986c0747e7f40e7ac2240d8f2043e48882cc0fddc53c0d1249bc0bc950cc02aea2ec0a40aa53fb3a51abed36f3dc049e5b5bfbf9786c008d6a23f4d2c853f65e2963f01ef32400b2971be6e5905c07459ad3f47ab79bf6b98fcbe07bf383f1a2d95beba09a2bf945750409e4480bf724370bf25301740b3d502be11ec1fc008ac193f38265f3ed7e6dfbe661783bf4e164bbf21dcc6be63586e3f5e4bafc058b5b3bf3f0428bfb804d13e19248dbe627b4dbf7924a03ce1faac3e1ad5763f426456bf87d53940d41c6740182808bf1e38163f11850141b761dcbe49a56fc01c5eb9bfafb60140128f813f7923babf5065da3fc4a1eebfbd9af4bf2a9381bf97da6e3f8c8b253e751172bed4093f3f66610840f81e76be615375bf957fa53f4c6641be08ef1e408b7694bf5d182c3fcc103240ad1521406aa389400a9b2abf60551c40af8e683fa4fdfe3fba2be3bf2cb811bfc2852a408e5be9bf8995cb3f90de913f6c97fcbe086554401669b03f6ac44740c6ac94c0909f27c04ab656bf3d1e6840360c52bee86a6cbf8dd59fc01f150cbf42e000c0eb3a4abf0e150640546eee3da5a2c5bfc16301c0601fde3f36a6debf602dd3bf872ff8bf9cd7bc3f9eca21be950f893f173746bf8cf302bf5d946b3ddf3a883eb1a039407876afbf6263ecbff1accd3f30ada33f5b6825c0a78e4040e00497bee7952cbed492f93f41759cbf7e77d13fcfdd563e4189f23f98d113c09bbf61405790edbda3ddb53f7bb8d3bfc85c1540ab583fc02bb1933e6e52e13f5ed318c0f98cf83e053e3840e87ece3fbda207bfdce787c0de5892bed6befdbefb1f3e40600e473ff740c2bf7a105bbf519a47c021e71140720e5dc01aa1cdbf4d1087bfa69e1340f96404c02d52ef3fa6d9643f03b489bf0495f3bfcaa9133fe0362fc01d2a283f4961e9bfba7f723f77de1740a8e40abf4239a53e50a6003f1217263e06528dbe0273ad3fc7df5c3f37fd68c008a22c4083443dbf0fda31c0fc6461c0104a10bf699b613ffa3761c08e9b5fbf3e515240ac794640a5da973fc8cd65bfb453533f8b05493c052b1ec04b7058bfe2418ec0b077d3bf29ca9cbdd62abcbf2aad09be6cb804c01d0a333fb7dd7fbeca645ebd2784f1be9ae0b03f2a328940902516c0e02dcf3e969850c003b540bf691f30c08df4a73f9b988c40be81eb3f679d94bf8258df3fd8d620400b7cc7bd051988bf6c59f4bf5a7d0fc0030b6b3fba57c3bfabc4e03f06dc433efbdefe3e7dce443ff73123c0073540c0222ac2bfad5e6e3faa36173fb94944bffbf8c1bd97dcd6be06200ac06bf2db3f972b163a10ca853f23660ec0e3da4540b2b27f40cc64383f11d1a63fc4b6523fa7e65cbf438f4bc001e996bf4fd11040c831a6c0ee87b83f32f90d408657ffbff092efbeb4051140c960ce3fe8d8713ff6cbe7be251293c082dd08400ee91ec0922cc2bfcc2c9cbf3ad47bbfcc8940be6747d6be21334b3d23f0b7bf7fdb83bfd99e85403336833f471658bf3b027dbeb91df7bf93ed2cbf3ac7cdbec69550bc13153c40853e2dc03627da3f4fd0a7bcb98632be4fb6c0bf1a685cbf8accd0bed5b6a83ff5d4e93f7764bebec91716c0fc56f23f83a6993eb81b13c0f8ec57bf02e2f3bffa89a5bf9cbca7bf4c9e15bf2224de3f14c4154016b1bc3fbd1f264050198c3f7978f2bfead706c07bc5cfbfba9f25c0939e693e2d1d4abf22de483fcb72c4be53f0bc3f286009bf8886fa3fa5bb06c09d7baa3fefd74c3ece380340eff1ed3f3348e8bfdd28a240a343453f23a708c0d7ed2f40c0a658bf83b861bf08ba2fc01bc917c08072b9bfb04908bf3e8727c0a3d796bf20fbae3f135462c01e3f24c0b34aa03ff2228abf21a7313f18e9873e79d35ec08201153f529c15406411abbf283248404b0192bfe464243f7fbe27404a4ed23cbf93d3bf86595dbf252db33fb14982bfe887d53e4565b5404a5211c09d0942bed79f64c0b2ba1bbe9911dbbed68f25bde7dae4bfb18da53f9d7659c0bf970cbfabe89fbf75b272bf84d183407daa3b400f2889bf2b4addbe87f5e7bf6732fb3f96986f3fb5f9943e2907f8bd4d795ec042a52abf6c44773edd3701c086de9abe74de14c0a9abcdbe0d509f3f4f9ae0bf9ce455bf978012bf2d716c3f08c1a2bff06acf3f3f5d6b40da8001c05d5b06408ed238c07090423f4999683f51dbed3e4e9d8bbdc04c62c042433840686fcf3f1055364052c6903ed0fb6d3e0512bdbe23a84bc0906ea0407fec06bff4d918c0b4ffdd3f4f0b81c03c77f8bf128f0ebf1657dfbff113e93f0c2c58bf53828e40c79ef9bfccb4f33ffe0f0ebe8177f63e64b5953fe50d7f3f0e535e3f2f6e223f19c6cb3d226f8b408b10e93e16d280bf1992db3f9971153f50fe943f601699bf724a0a3f092e70bf66982340063f2d40b7122840902d104093cfc6bfff625d3ff73d9c40070c8dbf470dab3fbe24bcbf423c34c00776db3f8b1650c0e2e04fbf3fd4073f5ad0ba3fd9b142bfadcde9bf748fd33e6c570e3fbc96c4bf19751c3fe15016c04e647fc0d804b13db2de0a40b6e20cbf8ea860bf287c03c00ae58a3f2dc5f5beb804dabfd698f23ec20a593ff0aee9bf0a169c3f213eed3f344744bf8fc798bf978d30bd2f89d1beb5f4b3bf077ab1be5484f03f668cf4bf3c6303c0a43318bff3e4aabf43cc8cbf54dbacbeda7153c0dcdc6b3f1a971bbedea36abf69a746bff6b5efbdce04bdbce8578ebf084ea73e364d5ac0c79b903f2ce4edbe8d0cf7becd4a46402f5a8f3f08d204c028162d402b6cc0bf17648d3e020926bcc171b43f89092fc0ef61c43f0110bbbf3aa3c03fbfdae3be33c061bebf13f23f96697740f5dabdbe8eeb4e3fbbce4dc0426a12c01599a8bf16a344405f68b4bfc4461cbf6874353f69678ac0fc1fc03f86002dbe3170e03f13cb6a3f202d53bf734b113f626cc2bd982efb3f087089bf96a4e53fc3c1424064b520c03b263f3f752d40c02947cfbe84507fbfead024409d6d0e40c2b50ebebc0b44bf71ad7b3fee3871c08d7bff3eb11808bf97643a405b3a883d2b4ec33d0436ffbef735cf3f3f2c5f3f4aa7003e50cc47bf96b1ddbf2c7d76be75b7c03faacd97bfa7b82a409412d1bf517ff43fe7ffadbebf5dfcbf6b0f97bc39158d40efdb7c400b772ec0053580401e6b2740e694f6bcdbdb90bf10576ec02e923c3f0c68ae3f386b96bf2868ce3f50e5133f13d4b6bfd6ba813de9545f402a7d85406d1459bf43ecbebe87cee73e3e9a1a40a7b204bd8959ebbfaaafc43f0168e63fc7611f40950ed03fc13eeebf0a3aacbf5903c13f9851d8bf0ca326bff667683f99ea2fc03349193eb5dd183f9c3d02bf176b7abf5872813e872d00c08575c7bf6a2ef6bf86ce0ec0daec264041f986c0b563af3f215d743f246d543febca64403d96f83fa8dae7bf75aa4dbf4e7032bfd5b5024069fc1cc0b1aa5740cc38b13fd03e55c02bc2ae3fbc661240d58007bf3fe187bf98d3f3bffdb96bbe9c1e48c0c13dadbe79ac3fbf9d951ebf633bcabfb49691bfa6f432c04ecf2a408bd905c09782aebfdbf20ac0df7e0640e0c5013fb12318c0bd09734000bc79c02d142040565d513ff47102c066b919bf234943bf6a4de8bfc9d96c3fc9c0cf3f6853da3f3c9f19407398073e2d00fe3fa124a1bfa37424c09e8a2cbc423d803f24d30c3e833a993f8517a0be931655bf451a0e40c0ab583f6c7bd53f85776cbffd86b4bf7e61e4bfd57c00407e088c3f5bb0b9bfbcffddbf639127bf8038fbbe98b68e3fd29d4d407bc83f40a5390b3f290a22c0f281edbf13b40340950bd73e688a38bfc174d13fd9fc4a3f8178194038be283e85e9043f97b656400b218e3f16aec1be7903ae3fba7443bfbba402bdcc3de93f929226404cc1174052e91c40adc922bf9525f33fa12ffabf7d77694079e3ea3f75a798be8969623f476591bf9048cb3f377726408e58a8bea45b06c0ea7a783e12e720bf0dc935bfbc72ec3fbfae62be896574bf056a06bf8fd5ba3dc606cbc07a40c43feeb51ebf25ed913f8abeaac02eb41bc0da6fba3f3f13793d95b4393e51b9f33f8cd94c3f09d5e43e29b1c9bfc6c267c0e5200dc07748f0bfcdb8823f3a2bf7bf1900963fa7863bc0892287bd997c673f2f8f2e402b1885bfd60987bf831458bf4e04b63c441c49bfb8fcfb3f28c195bf7fe38d4029d6643fb48f3bc0a1d9433f0cee7840a64688bf459fedbf9d298e3e712adabf9fad53bf551d23be9fbbfdbfe0f6e0bf6d90d6bf829c2c40e92578c003ab84bf5f0c3bc03d124340e9e9503f0a6eef3e869d3240e2b5ac3f415db73fe59e623f07b556bfca58903eab69f33fbbc9f4bfe1cd30bfe8a833bf7717223f0f81283fdb814540fbe2393f4518c93f4c20b5be83761ec0d34da9bf97ca22c00d851340db2580bfb21b6940592685beb2c111400e13b03e27fe50bfd21f114098ea24c09567e0bfcdd1f9bf8f72dabf4116d0bd718732bf323e36c080265dbfcedf8abd36ff08c03a3591bf6ee6c73f101a8e3fa9c6b03f3cd8663e01ab15be3f66953f79e6bcbf3385b4bf517a05c027b80b40a8dcb83f3b48d33e8bfc3ac04e44a2bf013b8dbfad52c8bf44941b40f5cb1b40f8290bc01a8326c06211b9bfae6075c0a1a25f40c07516c0516235c0acb04240c4f98cc070430ec0d51d4040e6a3a23fa34e19c08a3bae3e3b9b1440f14645409381b5bf6e8de13f4e25d93fe23946409d36d2bfc96202c0fefec9bff2b14140716f32c08bd12740a84ab2be3d16bb3f902b2dbfdd7a3e3eca30d4bf7841ac3dbce9144088900040aec74140af671a3f9bb108c0bd739f3fbf8e1e40310b554054718ac0f71f54bf78fcf7bf9ee964bec7752540168067bf4ac8d63fd1188abf65320bc0df32c13fe2a8ecbfe5c3debd0f3a4a3fdf4758bd5256e93e67a18940595ee3bf6bb5cabe427929c0b03c2ac0eca5a0bfe1a5aa3e4c9215bf00156fbf0f2f133ff1844440f25fbdbebdd031bdbe939c3f8c53983f99fc7d3fd6d8a43fd3810140083f92bf8a10ad3fafa5b5bfca3b8a3fa3e5ffbffb4ab6bf3170123df9e03fbf8eec37c04175363f4f4081c0cdbc953f2ba2b24026de7740a39f17405a13b4bf3ddc1e40e2ec24bfe730af3f85da1e40978403403ce48540f8affd3ff3df38bf86068a3d0ae70240dee39b3f08e02540c0b147c0bdbead3f6eb5843fcbf87e3e3693153feba7484035f632c086b7e6bf526b9ec066ae8d40590111c09a9135c0652100c06d19863ed401c63f40ea223ec351463fc93933be1ccca73e6f19e2bf690c433eb845bb3ffd687b3fcd82c93f60c996404c1798bfc5548bc06aeb02c0dacb9f3f10a505c093601140b7f5eb3f423989bf483a37bfe72589bfb4917b409559b040ea30413f7f342abfca6f5dbf7acb42c0e5dd51bfb2c66cc032da18bfb32f4dc04b8e90bea8d22fc0243ba03fb52a18c05f53f2be03111d40eeca5d3facd923bf23ce52c0926ce23f128396407ac132be88bd93c0775a25c0e28f9f3f25a1cbbf581c91bf93dd4b4000eabebe246036c0bbe641c05c4047bf7ad75ac0b5c151bf255604bf1055823f73620bbe46509ebf5e19513ef91e81bfbf05debf9a0622bf0f726dbed4bd7dbd87a0724074543140c1ce8b3fc15a384089ed8abeabb48e3e61da51bfb694ffbe25b32ac093fe1a3fcec9b33fcf6d7c3fca378e3fa6857d3f35c0c33f0f3fcc3f9fe941409bbcacbf33c84cbff7d4b83e4633a8bf3f9aa4be882d17400b4bacbe99fc96bf45f3af3ef689623ff107673f8754163f96f1a3bfc7ec453ef01a6d3fe781d23eb800ccbfd661b6bfb053633fe23b25c05d08b93ea3c950c0b11a4dc028f78a40667958bfbea437bf88479f3f9ece4e3dd278234009cf1ac02102103e6801c0bccfb106c0f04996bda41f08c080f8ebbfee83d73fb4381f404cccd7bff5389c3cce715f4066eef9be1199a93ffc8fb93dc7ac933ff379343f3d64a43f519d9540efe7483fedafaebec1b853bfb3e417c095c8533ea78bf0be775f8cbefd1b144060df2b40be8755bf1d06f73f5ca4cf3e5baeb0be08ca85bfe89525c04b4a10c09ae0b9bf11c6da3e99f7ddbe349121bfebdf0cc0a92254c01c116c3ff9dc0740a613d83f2eb3953e40308e3d62dedfbff15710be80747140499123408642f03fdb75933fd1a1903fd29234c046bb13bf7e34f3bf6494874035d7ea3e46de38c02ffe2fc058629f3dc555fe3fee363b3ebe4dfc3cba39dcbfe3d2523eaeb542be7ae456407406b63fca3d1540acee0abf86504d3f35374e40daf036bf925f32409dbc4c3fd567fa3f324498c0b1c82fc0bd14dabf8bb3de3e0308cf3edc4d2ac0d3e9713fb95a37bd952b1c4039828c401c7921bfb3402f40be360240c54ad1bc7e02e0bf4e74d9bf8ae1363e749ab43d0b7833401013a9bea7d3c33f91c4403e6f535c40dde081407c7e92bfe4a597bf04c70f40cbaa4a3fe09ffebecb4e25c0907317bf21999cbf13cb06bfbe6534c03a8306c0fb2345c0d01cad3f84441b409a8e1f4011f0c3bfe56f8fbfce504cbf142c4b407609ee3d0aa027407b6165be564656bf12300ac0c4bd444012065dbfce38f1bde29a37bfba70f53d4e62c73e916cbf3fc4a849c05eecbfbf9750133f12a26e3fbe7f70bf763aa5bf029efdbe59a752409e0923407a384bc00ddcb53f82d70040fbb790bfea1790bea38e4f3fcf98f03f9e06d2bf02f7a73f67392f404619b73eda952fc09aa46ac020c627bf5699513f6cf4493f5261834029b5d73f7180943f20c9d23cd66cf9bed33fa73f7d354940d75c314039461ebfe0fcdbbf07a57fc0ca80823f8a874fbecfa0b9bf53e72bbfce188440d72cbcbf1231f5bf24ff784077682f3f069b883f831afa3fe02b10406531833d547cdebfb1e0903f9be42e3faeedb73fec608540dee58e3fc32ba3bffdd33ebd02de83be30c66b3d81289440cc9d203f82ca0a3ee245f63f5061f6bff6dae5bff65008c05e94f0bf995cfdbf99204940c1337440697b8fbfb6ff8abf1d75fd3feb4a9dbfe3c215c069065a3f59f14840470ac24093cd8dbd59fe56c0616ca53fa74f04bcf7625e3f8105aabfa304b63fc1a3a73f1174033fa039afbf669821bf2f6dacbfc4ca823f9eee11c0c3aa0540db5c2240e7063ebd877ad23fffad903d3f4c2a3f04970e402b20ee3e6b7480bb53ec183fc4920840ee6524bf005d9640b89030bdb4c5b0bed9372f40cdb75040953058c02441893e6f497a4043ff40bffc6f04401ae13040e21841bf0e56a7406d30073fb4761640f50945bef23277bf882c1c3f9041d5bfef8dc33f572a29c03745a93f058646beaa423940a6e086bfd128f5bf062882c0de5831bf0b5a3b3e3891bc3e386316bf3382aabfe361a1bf1e21f33f84fda33ff04286be5a414b3de8861d3f4b76cc3f6ea3b23ff101f83e6c2d39c0d3496f3fcc213f3fcfa1713fcab01ebf5c4789409b4bf93f1bd521c03586ffbefdd17b3fe7e106c018cf8bbfcb050ec05672943f6b0b7440f0104b3f2142f23f750c8a40776e78bd0746773fd1ed02c0ee09bdbe58012b40f2167a3e8c1b61406e7125bd95f01540572e49bfe4bb25c0796c893fb25e014054b6c83f52b57f3fec2cc2bf47edff3e196fc13f04e524c063da95403a64113f85020040283f454013130bbf73815ebe94bc093fc3d5a8bed648aa3f20bd9e3f6cd601bfaa9765bfa1b161bfb080973fbf8e00c0a85b0fbf37dc50bfe915603ffc7dafbfc6af203f60c3aabe886921c039cb383fed01113f7dfc013e408b56c072df0ebf8adf493fc0fa17c05449fcbfb86da1bfb08d174016cf33bfcad0293ffef80340068d0b40b111573e54eebe3fab09d43fe1ebf4bf7697c3bee39687bfc9b9ffbf642de7bd1ee585bfb4ad44c0ad88f33f104c08c0898d4740ccfd69be312a363e5dbc943e538a93bffebfc3bfc9730340526850c0901bd93f8cf71f406a04c63f1803fc3f2aa98abf733a45bd3948e83e576359c08f19d13f1a14063f0c16b4bf337206c037c6b83f74e2ae3fa57bc13f1d10c5bf9c14d2bfcaacd53f337282c076c01140a41dc83f889f03c01aafea3f34d05bbf4a04c8bfe1ac8dbff3b21dc016fffb3edcff00c05d62913efed1d13f30840340d6d5b6bf77a8ffbf8adbbcbf92a830bf127a1040db8297be4e0c6dbc693722409dc309c0901c6a3ffbc450c0ac1921406df72e3f1f9040c05c835d405094ddbfe135ff3f7e54323fe9b6b03f59aafa3fb7e737c0056d3b3f843931c06a64104013b4cd3d83a465bf50655b3f42301540d5adf33ff4d1153f1404babfaf9444bd2394c5bf4e61f2be1bb635406d7034c02e12803e0945ad3ffa631dbf3433babf256c0fbf967f9b3ed94dfdbf4c440f3d9ab80640df758dc0db6760bfca86f4bee9bf5e3f506fe23f89a52e40e0e08f3fea674f404f8149c0a0fc9b3eb7c584c090f934bfc299c73ffc756e40184064c0268306c0075b1ec0e04af33cf598023f19bd35bf2b2bc5bf11a1943fac40abbf5f5974bd4d8f8f402d0d9dbff5c7f6bf9ea71cbfa1f05fc0726dfc3e0110d43f27ea03c068e25640d48c59c06abcae3f2282b93f6cbfd0bff42f6c406545483f6a942a404e64b0bf53aa82bff3568fbf165397bfe94b01bf5680a6bd93fc59bfd679e13f8c0c4c3fed97493e50d7a63eccfc82c0b52321c062bc29bf6f0fbdbf8a853bbf1612244070a6cd3ffc75bb3d6e058b3d3f97e1bf8dc3f1bee8d9b13f972f9640c1b0f8bfe52fa73fc40b5abe5505873eede5a6be2e5fc13fefd29a40a7b93f3e57d91bc0279af5bd42e1f03ef12199bc994e84be3f6c314052aa35c0d9090c4053340e40d310864061d7dcbeb45c2bbead187640344d5ebe1cc7893fdeae35c0dbca92401c525e3d230ae23f519b9abf74b4703fb00804c0114e16c0cb226d40567699bf4c9652c0d3f779bf80f02bc08f0884bf8b8a35be858b30bf41726e403b7d543fe82aeb3f87f6463d356c00c0cb7206c0105ff9bd1b0e713eb98195bf46e180bf66e5093faedaffbf2bfb1140a22d3ac0c65f0b3fc3869fc089adadc02661e8bfc5bb8dc0f09e703f42e2523f35799ebf629c51c0ca8218bfde8bfcbf44e6763eafaab93e4042024035683240ccc80240a9746dbee7a08f3f910ee83f63a505c0ec1e2e3f62cf284068f23f3f6fe0a23f061dcf3fb96287bf726b17c0ff55abbf62170ec01989fbbf9a96a44078ab06c0c64caf3fcd6ccabf0ae5993d355890bf19cf4dbf9e43d1bfad5e15c0e48d323f1bfb8cbf96a3b63f3aeb65406eab3f40f3df483e0034aebfdbb474bf592f8e3f6903443e4db2c1bde619f13feabddcbe372ea6bfb1591640587121c0cdbeb9bf46eb1a4094daba3f1c6f84bf0d6b5cc0befaf03f703787bd8633a73f4f8f68be4750edbed1e062bf9d8446c09cd284bfbb572bc04337dbbeaba081409289853e93df8cbff910ed3f924036c05d1346bf8f9a5140487cfe3e5f5511bf800240409788c2bfc83e193f5302db3f3a3262402ebc1740eac0213f3d40efbd400e36c0ea71073f6ffc40bf516dd13fba72533fe1bb48bf5ca98fbf153bacbfa2ef5b3fbe6f06c01379d03f347587c0072a1540c23273bfa2281d4003db1840c6a9613f35a9ccbf0ae0543e3993ccbf9e41b140bcd00740689268be8c6937401c19eb3ed479a6bf54982ec02cfac23d88f52e3e53a12b3f4b962abfcd9ee23fc17a9dc02e4ccc3fad0662c0c43ffa3f22800d40f49fc03fc86728c04b61ea3f925fb53f43037ec01abaa1bf06a4c4bf845e84bf92798340202810400176053eb54c38bf777146be574f1cbf18a4c7bf73cf5ebe9b6d0f40cc52cd3e20e6b93f516eb8bff7a0383f6e847cbf0226604016428e3f28b6a3bf88c01b3f0bf04bbf98ce84bdf25e10403f590bc013be6f3fbbca6f3fa75bc33fd567503f3f25a9be24cf38bff2e31c407f17a1bf3d7611c03384733fb725453f20f4e9bfebeb873ff3ca7a4020a580bfed2916be401488bf144157c0dc36bc3e5e0a0a409d1a46c00f9414c07a53bcbf560e43c0d861a43fee50783f3634ac40a89c18beb7543a3f1b72e73f3c28213fd9d55c3f526c3ac07277ce4010fef1bf44ff78bf06bd48c0b3cc3bbf53782f4063b6714087ae8ebef3bb093fc0c49a3fbde144bf2b5edc3f2f6c89bfee44db3fb1eb4ac0c4734d3fa1f8f43f4f5b25c0353cb0bf4bb631c04c8284bf470a45c077bd22c016304abfd50778bf0c878440a9e6003fd39b04bfbdf046bf8215bbbf80e1b4bfb8bcb33f87dbd73fc438484015206ac0c13f49c0a0eb9a3f7d0b9dbfe043a03f7db85640c76c7f3f8a2b213f9d1cfb3f20af36c0dc6faebe592d3dbfeaca4a3e2d8c8bbf06148540572d413f5bcad6bf796635c0c3f985bf7be7d93f58eb98bff1577cbffbf111c00b3a123f0e80e03f20aa9dbd9fc89f3f6c59853e84a5a13f204a5c40522938406a4f84bfa65d1a403b9b463fbab2983e758c223fdfe8dabd3e1f3bc0fd847d40f91185bfebad4cc047ac833f57583540de8e4abe5a85ed3f7b3ad1bf7aa2f83f2d9602408d0ce5bfe792483fdc8aefbfbe9fef3fd09fd5bf6f96b43f6f2e79c0df73d93e8ea35c3f5850e5bf19b93e40a6b67740138eabbf63dc4bbf93b9a73e1ed6a23facbd003f6531dcbfcc05b6bffe82a0bf64c242c07676ad3fdb5518c04ded58404d95074011dab5bf84872fc0a8f5e6bf576951409f449fbfa82cb6bf602062c074008dbfe3231a405ccbb1bf0c933a4076ed3fc0456638407048f93f7ff1e83fff574b3f23aa9dbeaf8d14c0737f2ac0d4a7e23f1612243f68586bbffd2f17bef74f4c40744e633f3fb9014077a1973fab89ecbfe184253e221ea0be4ce4d3bf23a061c03b2ffbbfdcb439be1ff55d40c227094026cca13ee73c99bf12538540e37d38bf00ddbc3f7e27b73cb7fb1240adb1203fa3252e3ff5f6963f7e44893f209869bde7cca53fdd940c400dd5d0bf60761a3e6fd1a4bee6e431c040659740cb8bc6bff78533409cd8973f5e2b05be32515ac0da5a98bfceb7b73fc96eae3e6decaa3e0f74c93f6b809a3e2a0afd3f7c87633f0dcd4e3f6b13d0bf390d78beb1a90abfe6271abe77214240c9c3d83fdbcea6bf8f2c8ac0b0fbc33f1ee633c03febaebf32d010c06ef5d03eb7c4723f927ad6bffb29ae3f03b2a93e2bc47f40e7d5653eaaf1893f195cd8bfe27e5f40a6e87bbed28b923f3fbef43e3b40d53f0ad789bf043544bf02c3b03e6f2a93bdc56ac13fd18b173e8ddcc0bb6568b3be35612a3f7357423f044a453f0b4e4c40232c00bff9edc33f848821c091f7ec3f2fe0413e7a1b5bc03cefa5bf9db138c0629216c0d736324021551bbf9a952740473ad9bf0b1ed6becbb26dbfe7638c3e2a7fdb3ff8b789c0618024bea20b3c3f5d8d90408632484039170f4062a96bbef9b16b401ebfa2bf356c33c0a43df9bf807b46bde67679bf79a30740933af83e87aa6a3fa5f37840bef0a1bfe07e3cc05396073e92ea09bfd255bb3e921b98bf1b1a21c07b3b983da5d8a1402bbe743fc7f6ed3f010021c082be7a3f7d43953e0300e33d17b2f33e479a1a403068094080070c40dff01540a7d81b40f44cff3f33d89140fb51663f0f62fabe10a051bfd519c4bf90e85b3fa2a607bf93e44b40123bda3f18d48f3fc428fa3e58467bc014c3033e8591443e9137a1c0f8f2c6bfe1581c3e3a24a1bf1964a1bf6e79f2bf93e00ac0363f0740c7e58e3eb45337bf90f35a406d3df2bdb09f323e2e593c3f8bdc30c0cf8a33bf5beb0a40fc664f40d144873f27dd53bf06b016bf2df74a40f2fc82c078fb873efc71653fdb9241bf0958004054e50440caf185bfeb6d77bf744933bfe0a103c0a17644408a390640a46885401ce5b63fea69274006fb1ac0e9428b40b3962fbf96ba1b40c9f3ff3faa9585c09e85673fe948a4bf87e9cd3f19ced83fecc9993fcefa8e40b9179b4082f9e5bf0f08ed3f49fc9f402dae9f3ffe67a5bfc92a9fbfe44229404c6789be0e3f424067ab9a3fcf5cc33f4f9422403879a63f9997e93fb14ea13ff4662ac0b230ce3ea261873f42f11ebe42a615bfff271940a7fb36c0"), dtype=np.float32).reshape([1, 3, 32, 32])


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
