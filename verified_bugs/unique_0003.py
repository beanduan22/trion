#!/usr/bin/env python3
"""
Bug #0003: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['layout', 'resize_cubic_halfpixel'], ['layout', 'reshape_rank2_roundtrip'], ['constant', 'redundant_reshape'], ['broadcast', 'log_abs_eps'], ['constant', 'cast_int_roundtrip'], ['normalization', 'variance_whitening']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0003.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0003.onnx")
INPUT = np.frombuffer(bytes.fromhex("08f56640816676be63573b3ef76e5fc0f71a3d40b1b2574067235bbfb24d6ebf810aa23f7022dabf333102bf169ca93f56dd4fbf4cd6294081c0673e86d264bf4a5c4940369761c00838aa3fcc8e6f3f2d08f7bf81b094be72b7b93f8c0c5dc06ef1aebed749bf3e9a87303cfbaa4d3f0346b73fba8e35c0ab76c23e9df68bbf37915d3f5da6393fcb93224068e829c08eaceb3e6ca119bf51f1cf3f4041ad40cdfe8fbf24330bbffe677d3e0fd61ec068ed52be461a4b40da9c3f40de9023bfdd79e23e34e68cbf7b1be5bffc68144001a2d93e6d9049c03e51fd3e1513194072118ebeedfcad3f25d528400131b4be50a99b3e9cb4da40e21909400ba4febf3bd4be3fa0319bbf9f57cf3f516ba33fedad433f501f48c01790fabef2d685bf3d9fdf3f008c9ebe5d8c38c028964340695d0fc007c1e53fcf2877beb205773eca820ec025825140622f8040f7a03b3fda196dc03a35eabf16f52b40294283c015911bbf26352540abfe893ffda96840ef14693fab185ac0dcdc463f8c3cb23d1bf977be129e6cc048c415c0df21f6be90f3434007e786bff7955040c687513ff977c3bf875c993e797729c0947cb0bfd26f0f3f8e88a03f4fcb9bbf6d3247bf365b3fc0c053b4bfc6abcc3e51a908c0af62153d839835c0887d59c0b5f2b3be5638dc3f2ccd613ef965f3bfe07c983f529521c0cb3788bf2faa32c0336a893fb67429c02b15174033510f3eef55093ea2a3834099a4674048c63040bb42753feecdc53ea134ba3f62c1273fe164d8bea79cf73f5305c13fe271e8be0eb19bbfd5ee2d3f9510b9c061754bbf203047bf8d68213f578707c0b3ffa33f82f957bf064eab3f85a631c047482b3f0326adbfa4d1b6bf5b9cda3f8153e3bfe69907bf832369bf0342d03f34305140618bb6bf982decbff23587c0ab257b40148e94bfcee50ac0e652853f7f71023f1975b8bfaf42a7bf8afaa7bf1209d03ab4c6f03fdfa10ec0c53493be625e5fc0d2895140e741a9bfb4013d4085adc2bfd157cf3f996f87bdb673703f7d74ea3fe0413fc08cca13beb601754016b766be87eb283f25d257be3dc93a3e6cfa5440a2641ec08bec7f3fe46006c0c9f37f3fb00aa43f63e6a5bf171b1b3e87498a3c6d88a3c0144d05c0de43fdbfae144ebe9b19433f63b6d3bf7b5c47bf75d058404accc2be06ddf13fc6c044bf3fbf483f7f79ea3f8fcd5e3ed1267fbe149d0d3f6289cb3fc40d983f9f3b943edc18163d802fad3ee5f2e5be55cd0bc0c1d5cf3fbb3e5b3ec607d53e96393ac0b0834cbfd8f2a7bf5152d6bf18a02840c7d300beadb836c0b81cfc3f82ccdb3ffff50fbf6838e5bfdebd923f37e78bbf7769443ea231323fe01c5ebfb96cc63e567018bfa578c63d4587cabfff5fd33f4785c53e6510b93ecff950407557a73f8fb8703eca0e4ec0675d36bf6c3718c0cf64b0c01d2ad2bfc29edd3fd6331dbf657455bffc8e8e3e3f0204c064086a40ee32b03b06f292be607dbbbf873a703f4fc58b409de688be0970b73fadc123c01dbcad3fa84bbebe903c6e40d1945940b3331840e681463f78d3353f6b7a9a3bb0a194bfaed0cd3ed9a8bdbfe927e5bfbfb559bef45536be85ca8cbfb556103b90ef5f3e1184babf44dc993f8fec1fc0718782c015fc163fb3e0433f333220c0e47627404b1b0640393cfdbfb003ae3f98088a403e8fd63f03b666402e335f40b4d98b3f25052fc042a7ae3f3f44aec0905b0c40b6192e3fa22a8940444a0340d8943e3fa35b543eec6fa4bf7e00883ea540264085b6bd3caf07cdbe36d3e03f9caa33c0151d82beb85d8d3f1721283fb8582ac0c8bf3d40760b02c07ae9cd3f5dcaaabf64ef88405933b4bf56f3a5bfc0c3dc3ec7ae8e3e5b222cbf96d9883f16d668bfc66d83bea546a4bee714c5bf23935ac0f864b23fbbe0953f423c943f677f90bfa39341c0d63987bf8985d5bf58fcc43e358b9a3f351d40bda3ed003f2029cebff3919e3f825ceebf3719f0befeb492be7e9369c0e3c41dc0dc35efbf0bf98a40596ad03f6ca06d3f523a1140ff913a3f5a9b8ebdca9b03bec58caa3ffa12843f83ad01bf1f5ce1bf4fe7e63effdfd83dd7c754bce89746bfd37147c04fee93c0cfec093f5f1837c07d856e3f817d33c0f1d14ebf15440dc0337c61bf8280493c9428ccbf888aed3f5b1d1bc05712cf3f35d1e2bf3e0e5abe0e15363e128738c0847c97bf962e013fc9baf93f4981dd3e9d35f8bf7516124025a3c8bfd659ca3f1c9a98bd324219c0818594c003ab99c091666f3eae1a97be40a2a03fd3c4893f02558dbe6321e03eb1c938c0e8a5d7bf8797c63fdc5e404032cc3cbec80e1140debd963f04ac3640edd32dbf0b1c0fc04e57e3bf13d68940534fa5bf07baa43fe4d1183f1370a03f1ae6e43f2dd440c044ea46c01c2c693fde57123fdde00fc0c66153c0e7a68ec06d71ba3fc9092ebfd68a14bfe5040e3f60efc8bebacc743f83ad033f1fb50640e31020405e380940ff0d26bf199a1d404ebc21c0a939cf3f9c632dbf8c99363ee3a845c0925af0bea9f7fa3f04ca0240e0d5c1bfebe0493e14901440e004bf3f2850cdbfbff37340cad658bffa80a2bf55d983c049d9a63f0e4fabbe675582408c2f873fece5323f1cfa28bf6bc3bfbf08b2083e7f0e9fbfe18645bf8a1d16c09cc36ebd157731c0d5032bbe5c314e40e5640f401b44e5be7964f53e5d2f56bf8f44a9bf622a1340f180f5be8c44d0bf296947c0957501c0cda3e03e6d3a3ebee462a8be0949ca3f0e4c77bf95aa0cc062e6c8be2e8a3240f41d163d63eb0dbf43f785c09f0a8cbf70a0a73e6a06e73eabbc2c40a80247c013d206c0060b3bbf71f996be41c6033f93cca03e08a5873f457447c0efc406bf6a0e4d408d7a3c3fbcc75c3fa6bc804077a7bc3f512eca3f576787bf61f733c0f81d37404896b43e17bab73f73bfb63e4be7cabf11d64e3fdc230e40b8ed0d401a4e05c0ac58443ecc20e13f6fefc63eded52a4042fa36c0a9411cc0046e7ec0a6954abf7d243d3f3ceabf3f8c8128bf2ae55abf54f5ba3f29b4eb3e23e33ec0e00ed6bebccf9f3e41e4edbf4a12df3e879a513f36a1b4c02d614abf9faf1040e3918f3fc3fd2b404c0ba240bf5b13bf1ffed03da4ff6bbf098e17c092c1203e92d923c0cf18cfbe785f8b3fea4905401b6a043f988f923fd9074ec0154f9abf3f1210c03762573f6b6ecdbf4f937e3d850827bf42e3ba3f8e97aa3f09964840038b8cbf5ad24b40c98fb1bf27b98240df3f1c4090232e4002fcedbfbbef71c0fb441c40020d8e3fadb1c8bd894b67bec034c43fb352b1bfcf60b2bfc7a1d3bec62ffabe9fdf88befbfdb8bfb8bf29c08e90afbfb99bed3f403c16c086396dbf36e4073f4f5bf53fd8066b3e9ff5183fe82e33c0dcdbf63fb91331c087a7403fb1c277c0077f7fc05e8c5d3f7b16e43ecee4873f38eb763e4c4547beb2e60540d0ae8940c64f9c3fbd215dbf650021be557cffbfc99bd7bf5700ad3f1f966f3a9f6598407ce1e13f7f3ce0bfc769debf6f13153e185886bf41c48dbfc025f23e14c93ebff33ba93f55d434bf07a8973e48512cbf5f10c8bfae999cbfeb6dfdbe991d7f4039bc2fc05a31de3fc25c23c02de15c40bf0eeb3d3c0a523c5c42833ead02a6bf3c381dbef4330a3fba9df1bfcc569abfd53d57400829d33e37e718c0feef8dbf1d0bc53fda38943e0939d6bfadbb29c062a5793f2c9bbdbe0da1f33f2a30653fe1604d40b9c7e2be37a867bfa8b91fc0ca5f913f24d1093f7c6c59c0ab4c4340f48d2640bf8053c04ffd2dc030b41b3c10a9e33e718f384069210740875337c00ec85f40f1f62bc05c5c1cc0f0bf36c05e8ab84012108cbf95dc4d3f6711703fce676c3f5df9ad3ede75813fcac48e3faa5ddd3faa2e0dc0d33e8d3fd73528c0de6a774085a006c0918a113e6b6b53be78e7abbe0bd36d406003723f04eb41bf317802be78fd27bf04f55e3f22bd5040811468befd6658c0586ab13e511cecbe3b58d43fe5aeff3ff92baabf7166963e8bc494bfb1d79b3d8882ee3e65d143c075c965c0dc0c7dbec0f1dabfea84e0bfdd1bd5beb2f27a40b0b2a93f9fc011c019fe53bf270c24bfdfe7423f6066f13ee4f680bf167fbe3fb2c7c13f6b5e4b405bacc7bf20d227c046e71c40fc6cd73f9176bfbfc818b83f40bac83f16e3a9be18449d3f6f0fdc3e48d71dbfbbea0dc05c5c09c0e89a55bfc723dd3ea97fe7bf7300283ffc19ecbd659f31c0129a0dc027ea254017844a3f15a7544091ed2b40fc5d0340781016bf4851ff3f0e0a9dbd7e86c5bf5cfac0beb24cec3f4f13b93fa7e7ef3fd3bb0cc0d83a00be23050bc07b64c3bfaa6642be6006163f3b08c9bfc34a4fc0060604406200cf3f4f2983401cc46a3f8ef48b3ff74883bee8295bbf798ccfbf08261440a8b8773fe403853f9285454079b3a33f9f53773f23c951c00a2ab3bfb99eee3f911d2e3f68f02340cef86bbf779ad2bf62995440527cff3ca47d3dbfe7f998bf2776b73e747f504073767a40ba3f37408a3926bf8bbfc0bf5ecf4d3e85ad7cc02e669f3e7dcb10c0173008bf8ec51840c6553240fa5adec0fca633bf7edf38bf871d694099cf9abf455702bf2ab1e33f0ab1133fcb55ed3eb0e084bfe4e0993fccc4f33fe1936540622bc1bfcadfa53fdc492840328e49be86bf104032bead3f4143a23f6fda34bfa26cf5bfad1b163f8b51a8403adfc4bf8b33d63fcf98b0bf4219efbf96365e3fa1bfa3c004db903e770af83d813ab5bf56615cbfd063613e99f9163e221400bf37ea9e3f73af1040869a12c05c8cc3befc91e8bf7e676a3fbe8f5a40cee680bf1cfe6840a00acbbf47a7acbf7a3a3ac0f49a93bfb04182bf7d329cbfcbe99b3e2939823f86d16c3fb5d605c0e97b11bfa2435d404744aabf14719d3f3fdc0ec0fd00523f7825c23faae1983e60d5413e97ee7940ce0de73fe5bf9abf028305c07f4ee33f94031ebe13cf213fce48854070e32b3f4c05a140089c74c0a8104bbf116e1040ccde3ac02d4aedbf945517c0094f41bf006a06c079e07a3eaf1e5c3f63f42c3fb94236bf23db58bfc72d353f4d5117c0639a28c0b6f71a40c4afba3e337f01befe0e0ac066a114c08702f8bfdba774402782e9beb45a13c07c6233c0af63653f0f851c3ffb2da53fe07552c05061c43faad1a23d70873d4099f3cd3fda6058c01451dcbd89c2193f1c654e401478f2be18840ec081d045bf1250f73bc753423cd67f3440f5dc1ec00efc22c0bb7dd5bfe8d1943fe143b1bf7c2cbbbf567f8d4082aa763ff07849c0a257fd3f40d51940692619bfb0b70640c7c1dfbfdb5325c0fe25f33e8c2a12bf3d7d7b3f573a71400e362540953bfe3fdd71f23f1cabf9bf78eb94bfdc03213f93951cbfff4e2b3f729ca9bee701a4bf3abe36c0b82cc13f77730ac0ef229c3e0641f13e088110404500233da957814001fa9940002632c0a86f373fd271a13fe3a732bf2ff5813fc6ff2ebfba57c9bfe1b129bf7701aabfd5b36f40aa2fbfbfe56105bdc4a7e6be7dc45740b17c1bc07ff466408109ea3f594bba3fe6d9703f02764d3f5ecfb9be70de764004e5e6bd05de23c0b22df23b0dda2bc01ca26cbeea967d3f89865abf9341efbf844b3b4033d70140f10026bf2c1a85c09584efbf834f54be37c4a5bd1b1341c046fad540e2eb153e4b24bd3fa90e09400cab0cc0053e82bf8cb11e40c0b03d3e6f72a93ef78f58bfe94f8cbdf504efbe338beabfa1ac424058bf97bfd676b93ee9c01c3e86bd833ef26cde3f5fbe114018f639402712d3bf7d48e53fae5f3f4017689c3ee9c9b43faa415940a7f81c40158c3cbfe1e8fe3fc87994be15cfdbbf624206c0f8e8b83f79c5843fb0e71240d0ef2cc0459439bff7618a3ef62d993e3ca67d403b92f43f33c78abf047613c01d6d20c04259683ede1d32c0826f29c0676edf3fe87752bf945c95c00ba91e4049c0a53dc26437c074ad8dbfb27d80bf2fd322402b0e033f94db6a4014a4a13f1ca237c076c604c0cbc714c0aeff1840998201c0dd7c48405e50e8bfcf21f53ecfb664bfc3f7e03f624289c09a42fd3f441084bcdf8accbfdf0578c0bcb1f7be50613a40f79f1440d3d199bf09f087c0e6e8403e31d14ec0afe99e406e57cdbea4b33640fd6f8abfb83b333d670a26be10394e40f5b9873fdecb0c403d47283f52fe03c0f69c0ac0098f4b40399f79bf6e7001bfd43d87bfaeba403f6cfd17c0b77335c0fb8c5840f991833fd17e17406bacbac0625019409e029f3e96ee7ac0f483854096353f3ef7949b3fc74ebd3f16ebb53f3550f8bef206e03fff6f16c075424bc0367672bf5f4d264054d08ebf27dbac3e3c1e82c089c31b40538f60bfe6b9a73e1f9f1940083673bfafb18dc04d7562c09e5316bfc8348940662469bf548521c026f5c9beddbdc3bf5325e83f6cc892bf00096dbf760c2dc0a7a3d53f35a200c0aaca88bef2137c3e02777b3f0f0552c039297b3f7b1b05bf9a0d9e3f9ef068c0dd179f4057578dc0a6049fbfd8132bbfbe7ef13f5c241cbf4148f93ee80b384005b103c02d9a85bf1ca500bf40e3a63fd9f535408f39f8bf16419e3f99edcbbed363e23f1e5bebbea2cc903f1ccd6f408526343e6b7d71bca17b9c3f3c0fc33f988a043fe51d1d40fd5c54c0e537b53faee744c047c59fbfa1031bc0d2af1240dbf4713eccbc91beaa0832bfd7c4333fc64083bf626459bf80982ac0877097bfa8992d4024736d3f8a579d3ea69878bdf0004bbf8a1803c03c1f72c08516153f50b7e73f1b8731c02ced85c00e579e3d14c605c007d376406dc2803fb2e9943fb81592bf94b2893f6452b9be7fc62e3f525d7abe99c44040abf67340dab13b4035f659c08e41133f039c54bfca419b3f68e004c07e07f93fe3aa383f3a97a13d1dc6093dc086473f755c043ed16837c0ba520e40961cc33fe5d4a13f47c7113f76189a3fc3c76dbff6072bbf15de09c085f4453e5c6eda3fa4f42ac02e0297bec56c793f08a84cc03aa00e3ebecdc1bfe1d90c407fe8ae3fcef5033a981474bec32a17bececfdc3dee0a21bf2fa99fbf28d015bfbe0207401f24f1beb2667abf82262dc0dcc547c01b7e28c0398a53bd46bd1bbf755d74c0a060b33e5284fd3e6087693fceb0ddbf0f851cc029a91e3f7025a8bf1d716cc0992e2b407c769a3f027f6d3eeb1090c03da0e43fa703cdbea49e1b406a4821be12f8813f0ee152bf6d3f3bbf44b2b03fbf2731bc57062abfa35f88bd50419bbfdf39303fd93751bf78260b40db25e03fc763e7bc02625a3f8ede1d407610fcbe2cf1ddbf99b07ebfbea3c9beccbdb1c09de2c7bf5d4f5040270f8cbfac7571bf3950733ce6fbbe3fc20831bfcd093040da35a03d2cee53c03e2a7cc0c46eff3f81318f3fe2022fc05feec640738aa1c05ed5463f92a646c0e90a1840efe9d23caebb883e312be5bd594cd5bf2d36453fbaf6a7bffb57a33f21aedb3e3e18323f9d265abe6f7416bf031239be1001863f6585b4bf53372cc08da5cfbf8e8f20407a3b0bbfba59fe3c8e7b5d3f8bc738406ba12abea59381bf33a161bf0a3c1c3ff4777a40f627c23fb8b5143fa38606bfe5064a4077f0353f5a93d4bfbdac263ffa025f40cc34883a29fa1b3e24d15cc02ccf8b40c94b9a40764657bef3070140bce80f407312383d0f1821c01f0d26bf7e8a83bff48968404bd3554058bb3dc0eabdd4bf681f8640b820613ef0bb79c0a98a1fc0a5d6f6bfce4401c050a5aa3fb85651c0247a66bf890495bfe692dcbfcfef7cc01d908bbe3f7d45c05beca8bfe518c0bd3e1437bfa97068bf3c2dd8be58834bbfbee362bf219b8abfb58206c0096af43eccbbf33fb208ec3fc5e013c0bc4885bfca94a13f3ebb89c00d46923ee666863f23cf7d3fe608a4bfc3201f40697eaf3e976398bf7fe502bf7d9d313e23ec42bf8b43fe3f10c033c0da2f46bfabcaebbff7a1c53f5f359d3daa8d2dc0128dfe3e9ff65f3fef2d4cc0a97925befd42d2bf3b77863ed2b436c087d5f8be86f795bf440ca83f21a7343e5b2431c04a2877c0c7346e3f448c353ecd1db6bf84a121bfaff4a6bd8857d53f15800f40dbd93bc086f87cbd9584b240899c503fc6d4c23ea7df1240814482bf053f6abc0a028140434ea13ff92330c0deda0bc0a945cfbed2183dbf5197a9bf82eb174014dea13d1a8b06bffa4c18401862da3ff339863e15bbd13fa1f103bfb24adb3f2347013f863435c0b16c0f40d4bb3240442d64bfdf56eabfad8aa7c0e4203640a0a8acbf2498babfae3bd53d6f783940902a543ee553c03f3e4a2c3f4318fd3f779b843eb594bdbfe2f1b93e566be7bfaa687bc0cf2a8b3f5f7ff93f605e81c0859e3b408d098c3f060e0a3e3c71003fdda00640a34d3940160b7bc08fcab3400d37ef3f82db3f3f55fc36bf2a636e3f44071cbf9859083e7a22853e9f7c54c07c756e3d155d2940848856c0f94d0fc0465f8dbf9661733fcfdf783f8bca0740ab215bbf059bf03e913b3d407a69e73e3d2c2ac01ff494bf3f9e1040706e81bf12bf23bfba62503e153976be301038c063d5c2bfede1af3effa531c006ec0740ab74c8bd418ff73e89ce3d3f25b86c3f2f8406bf04374cbed537b7c027889abf3ca78e3f25631e4071294b401293e4bfea4cc3bf8dc2493fbefe08c07d4bae3f145fc2bf5711a1c0aa1d0540e022243f0872603e5f39ef3ff17cf2be3c9338407be0a8bf662c4e400bc9abbe5bb610bd43618ebf8a4e3840409fb43fa28b9fbd059da83fa3839e3e582bc93f12ba53c06b2e5a3f341804bfd77d9f3f0a38d63f33310e40a5f0e43f26bd723ef819593f53b963c01554bb3fac5f973fb17f33bf9313f03cb77b4abfad8e57bf16be37bd6a4be93f9580723e0cce8d3fbee9c63f8751b5be024e1cc087e1a33e45dd33bfa8156840422a87c010460740837414400efc42bfdf043a3f0bbc163ece5fb63e7db204bff1d61a40bd5916408d072a4051b384bf7a74b83dfda102c055bab53ff093993fc3e246bfbc7fa7bfb73b0fc0906f97be53f6323de5fb4bc072e4d03ff7f4c0bda72007c03fbb0fc088bd3f40f90b00c0633e5abf29bad23f823498bf78a98fc05aae2f3f68c968bf93a8613ff63a2f3f75b6454019ad0c3fcd8e39402ec796c056b66dbfca051a3f8f096bbf2da3c33f0d280f407e081040bd4b9b3f2cc5133f7461863f27b42f408a33843f138ac93edd154dbf8a520c40333cc93fb2058c3efc44533f1474ed3e6354c4bec57010c02d9da73e783826c0786705402bb93b40107ab93d3d50294007f16b403e881d406af2703f4e1d0940a0fdf43f6c2adb3e916fce3fc2c49abf9b000d3fd446cc3f6377143f4287afbf36aff23f4baf05c0378f7ec0315c1cbfcfd42840097e1b4078435a40d54acd3f04f66abf183eafbeb78b08c0bb73e63eadb04ac01d4845be59f732c0dfab1840a097b63fb14afa3f26fec03fad12943f32a9b2be2ce5264035c0ff3f38570e3f257ebb3fe33f02bfc47d6abf143eef3f52c4ca3f5a3a933e81c580401460e5bf96f7b2c0aac724bf2a2d5cc0dc0158bfd18103c051ed16409d79c4bfc6d315403daa48c0217069c013d91f40b8ff893ee7382cbfdc1683bfdc6010c090e45440ffaf18c06c9580bf717e81c07766ce3fb0d342c0db357c3fe6bba1be10055ac0349a93bf9f3b6140fc461f3f1ed87d3f042fc3bf81740840ec98b3bf1f65b13f3c9b7cbf12faacbf36d3623f0d6ed73f83c9333e5638283f4ca035c04c8f06bf0681823eb1f4bebfc7c3203f9bacf0bf1392833f68e48ebfdeae71bf01dbd040c94a203fcbc0efbd32ab363f794e99becd8bc3bd1d978c3ffe5e1c405e9ce03fd5c2063ef05c40c077e5183effb369bf88e7c9bff47b67c0842209c0d8b3953f431d0ebdf4528040803399bff13ea2bf22d0b5be79a2883f048096c0621c5b4041020d405b4fb6bde9399c3f72182bbeae93aa3d938565c00d6930bebd3f6ec007cf21c00a4694c0f8943bc07f13fe3d8bd6f1be0353f4bfbe4974be153f49bf775c813ff5069cbf1ac14b4039cbc23f326862bff06b1a401f2237c00c085ec0a324453f2de6de3c9d5c0f40404311c0356809c0437156c019c9fd3ec73bacbdc4985bc03b0f4c3f7a23833beed51640a26095bf3365403fe8fe69bf66d005406db072bfe61f62be780afc3eaa2031bfd4bcecbf025426c07e7234c06a97debe641499bd2829a13d19fdc6bf575508bf7540a2bf34712e3feece903fcf5fedbc2c0aef3fc067253f2d0596bf7758b6bef27811be9bb2413f02382a3fa7e9ecbfa7a0f83e6bb700c0c0d524bf3304ab3e80ac553ddd2b6f3e6f14e1bfe4eb374096912240b5aa14c0dedf0bbfb3c5acbf600fc6bf249327bff0e0b43fc532fe3e38af9ebe08386cbf8a67963fee02903ffee66dbfbe3198bfabedc3bf9bc2ab3f2ced624097a6793ed1ef19c0ab7a04403977e13e7228adbf454ab13fda77aebe07eab13e854891405f2407c03e1e043e91002ac097d809bfa01f3040ca97b03fea8b09c09db66340bebc58406ba97240cb10afbfdca5bebe07b7883f06792b3fead221c0b9b48c40e98aa03faee2dd3e699605405f2d2f4058020440e2a1033f869456bf165ac43efe1404bf42821c401cadc43f971591bf740c1cbe6b8a44bf97a9913fa13a2840e21d1140936998bfa9fc083f7452a43ea85406c0a20b0340175fdabf3de6f0bf35f9dbbe2bf3433f71b247404e93eb3f8700b6bf86a791bffbe972bf0c1314bf13e3083ef58a374021e21e40ccf764409d4f5340b3d0c23ee7a680c060143a4008027ebeadafa2be6376e83f9aac0a40c8764d3f8aa9e0be906d074090ee563e0169c63f569b3bbd44fa44be657baebe9b0e12401a2ea23fe01b45c0718f5cc065be9dbdcbb269c0a857f93f1b6346c0664f37401fcbb7bfeed0e43de2a5663fdc5519c00472a73fa8b689be8f91104016eb0940bd3d95bf7276b6bf62342140f9f636bf90522e406b615ac0a2adc53f924ba1bfe34ebfbd18e193bfb6fbaebefe086bc08d0f9f3f78fef83fdf060040ba39a13fb74751c05e63c63fdd276f3e9fefcabf373e90c0f4f295bf28b7a0be7898ad3e7f6996bff1578d3de171d7bf0f0cebbfe4bdedbe603737408b3194be1c112dbf86536940f3cf244009cea0bf1b804640e3bb9b3f98713cbe7e852a3f04f2c73e1b92dfbe976054c08b4af23f786fafbd3555b7be99943dc07a841d402b287c40cfc3a83f1ebe8fbf6f272a3f6e660d3fa3a804c0dc35fe3f032fdabed66d23c0a675e43e158087bef178d53fd8647dbf81b9e0bf465d913f672986bf37bf943f20a61940dc65bd3edd6be0bf6f34a33fe02426c0a31d7cbf939b4d3ea739473fb082813e81c6d8befdf1eb3f8764a1bfec9400c07e627abf471da43f8862da3ff35d0940a560954005823f407a8f59c0806593c07ad884bfe8b2ee3fdae8903f5014953f4a52fa3fa6970fc0d99636c07ebe4bc03af34cc0551649c0266fb33ffca8ddbe4aaf0040ec994c3fc9b9a7bdf20c753fb983a63e518cbbbf89a21e3ea7f212403e422d3f2e270fbf3db0ad40f3782a3ee32f9f3ee51e3440d8a0ab40ce2a204016300c40ebe21ec01bc600c055e305406a5a25c0f8c92240e286da3f27e4ae3eef887abf73350b40fef084c0c90c01be8d28223f1088dc3e65dc093e521e2a40ce9a2ebca8ecd33f62c4ea3f83c7d1bf2478df3ff8254c3f26fecebfdf1b973f58eeeabfe66ca03f228b1bbf767791be67ee83c039c790c00b259540860e91bf0d52713ebd69b13f890b9a3f8b80d63f1347023ecbf4f73f3d8acbbffed8713ff7928c3e3a2c07bf32fb5f40746312bf3a44db3f4e66db3fb937793d3e25cbbee881183fbb4d52bf0e35773fcfeaa7bf75200cc04e60333f2d491040aed94a40ff06afbfb1d62c4010b3b3bff3c44cbe090c17c059de024048952c3f492ebb40cca7fc3fb7f01e40520d1ac07b18d7bf52731840afbf233e63c910c04980103fca5413c041652b3fa04605bf8fa4693fc1337640c45872bf2b5b6ec057618f3ef573cdbf8b801fc0aeb505408536bdbf5d75cdbe0b5608c0dbfaefbf4d4d9a3ea0ea4bbfcef9913f41a2ea3eb1c7e13fe1372bc02bc03f40df9495bef3ec1440d55fd5bfb2999dbf2dd4873f74633e3e93ac92bfda6f1040c0691c409dbd41bf280dd6be395dd1bfc966de3f7d0388bf95a0863ec12f6f4048452cc07c7bd5bf0172274069a5bb4057f7e23fe10bd7beddda61bea4d703c0db9755bf4da7e23e6fbd8340fc7635c085080c4009fe983f654b5fbfc3b01b3f6ba55bbe7544b33fa18a2c40f8274a3e112250bf69e562407b18a1bf16badcbf4a0c033d2b9210c055cbcdbdeac9f6bfd235d73f80bf5fbfb44edbbe7ac6b0beff7b98bd9c2cd8bff29b6cc021de383d0e55c5bf232777401a344f3ef9e4b23f1bb138c0a0e8e3be944a8040895800bfe4ad17c0bfdcc5bf3d81c43e3e3b11c0cef62bc07c5052c0965b123f060d09404de320bf3857b5be67ff67bf38fe883ffdba2640667c103efd3c803f17c14fbf09bd783f48062c40d2250b40a211f13f4382c23f63d3f6befe4cbebf253db03f44acbabf1789d6bffa4d8e40a1089b3e455428bfc9ce733f5c790e40e07c023ebc3ca4bff75875c0718110c05c563040772ebf3fb556a33f96c4d73e0e3d6440958b253eb07115c0214fdabeccd89d3ee455ccbfbfeaa8bfbdb243c07b4f54bf04430dbe1c51843fcc8de93d62dace3f48ab7540b8d03abf97f556c06e2bcbc052e53a3f5af50740d3f669bf5871203f5524fc3f610589bf9b3fe9bfc30f51bf69e64abfd846ab409b9529c022cbf4bea5bb3ebfbe28d8bfd1431dbf2e99a0bfc63703bf9e890fbfa85cee3f77dbdebd5795c83fb10eddbf14a684c0519a12bf99a01bc0dbc0983f1e2e12c0b9379ec0fecc1a4062bc683f34cdaa408e3ed3bf08d1e83f9b28e43eb04ef4bead59b4bf655b6a3ff7b303c0130190bfd70f2d40e9955a401b130e402d759e3fe5357440e5401940ad7a933eed017fc018e7f23f7b1ec9bfea46f5be416ba8c0f035f5bf258264c0041716bdeb8fe3bf26f1e13f4aac76bfd4de36bf39dd4ec0a58d1a3fb47414bfdb380abffbe86bbf86342c3f01b705402dac97bff0b4cfbe0289f03f83a954bfd573f8bf0e9ca03f94f4574093e2ae3ed9bc4bbf78190fc06e365bc0741985bfa8938c3ff7002fbf97cc08bfb23137c017937b3dfb03d5be0bb099bf9ad52ebfb2b7a1bf3e66ea3ff29db3bf443c23c04ed93f40e14bd6bf94a5473f786185bf1051a43f994f27405118d33f7dd386bfb74689c059b7c73ac6ada23f2705babf12af74c060599bbe7409ea3f7168ebbf6c21884032810ac0de8fc33ffde8a4c04ef21f40c7800940b2ae30bfe8a5b5bffe45f13e82f285bf96e0513fa7e972402b8149bf51c4c8be0d1271bfd919bdbf614369409820c2bfa8fe813e655884bf9b44aebfa690c640ef5a1040da3f24c065a68abfee24273b280147bf649d41c049a85f3f5b9cb0bf09cd5ebe237695bff26c4340b0478d409f3a0640cc765fc037ed2340dc260740e58427bff09102c07a3b77406d58ae40e690083e4bea32bf79a9123f019a0c3e1531ab3f617bc23e599d24404e3708c04006f7bee2911640a17a07c034f898bdb3ca20c0c719b83f3c9f3d4016c4ba3efc7cdebf53d7f9bf2f0f9f3ff158a6c00dd1553f8dca2d4036e4d53e806391c0f340f8beeaa3323f66195ac064f0b0bf04040e3fcdebd73ec4e0f6bf8bce703f92fd943fba8c9d3e76b97ac01c8cb1bf2a4713404f012fc05212874084d8d23f54eeb73fc21f1bc0807de73fe060cfbf2c2d83bc8c8913bfe51200c0c90907be702f0dc01f8ce23f1fa5a9bfd72680bf133130c0890cd5be5f5411400c690cc0e8189a3f45f7f43eaa71d0bc4c3981bf9c0d353ff2913dc0e61ce03e7061eabca2cd6bbfde113ebf86d9613f9a82883ff95733c0f9087c3f4216ffbf9ab482befddf893e934bb63f18390d40f51b303f1d9828c0296719c049e2373e9f2517bff1158fc047c424c075bf7fbf5f8c0bbf9c6d9f3f05f4b7bc8d3c3a40ce1140bff327673f360b1d40cc4128c01ecf0f3f8672893bc20acc3b79da3a3fa256cf3e56cd913f8e805a3e72a5484058a4e13e81f6f7bf0293623fc0028cbe9b5994bf1c04c0bf8b526dc038428fbf72c95cc02f0887bf941f06400dc1d13fbbd53c3ff41176c0eb57753fab636fbfaa993e3f4f98db3e0f7bd9bfaa582f40d88b3ac068c5dabe913d4abf29a99dbfbc959b3f525161bc3b909abfd14163c0faace03f671fc5be4ccbf73f593978c01104ff3f713abebf621f803f7b47bb3f31674f405c4a77c06500d93f5604f43fc7cd6fbe5583943f9259a03ffa9c39bf0f1be43c1b2a26bfbf6ca33f2529e23f7e69f2bf52d506c0ce5408c09d410cbf5c6a1e3fbf0332be02bba5be6591da3e31a4943f3baf9ebfef1cad3f1fc73dc0b1725340c8a1813f9caacb3e9e55693fa5de0b40e32f8440cee36dbf7a2a973ef780d3bebb626e3e87f0aa3f7e6a6c40e5faa3bf956d81c0bf86b040357d7fbf8c0008409a3ad03fcd370640b51027c08f367a4065ab2ec0d4c0c23f8c22cdbe69f15b3ec116a9bfde9c4ac0e1a89640440b4bc073ee90bfc2cd6c3e5b822140097b11c02aecabbe1ad4b4bf5b6629bf4ba41240d8d1a73e7e4b9dbf8e31a33ddf787f403b8eb03e72860940e74b13c0761443be65f9833f86e7943f801cc0bffa9268c0a6678b3f6613febfe8d1babf58581ac07923bbbff400ed3f3a2ba0bfd824f43f5e5543c0d1d7e73d54b0533ea525ea3f6e1d80bee2e7653faf6bcd3f915cf8befdef3fbf8af93c3e803afbbf6f510b40556512bfe17653bf853883bf79969ebfc726073f369f63be901c863f40bc074001d5bebd1260504085bb35409465b4bfeb9d53401b9a17c0588d8fbf4c4b843f79033f4090b421c0fc18babd2495b93e476f4140c17290bf71b88e3f50f87c3e3861873fbeb24e40eefcdb3fbabc3b3f1ad9b4bf7ab4aabfcbfa11c01d6976bf3ff90a3f5610b13f57a2a33fa77c303f2e69efbfa19df7bf82fe4fbfd3bf82c0970467c0b597bd3f3826f2be4b3c463f494bf63f4af8d8bf5ea30d3f921b0ebfb96327bed581af3f1f3382becadb22c03ea7c13f777eac3f14590bbf42cf1dbecef80fc09ea085bf458113c0ab882e3f4912d33f1a896d4025805140047de5bf7bfb8b3f79d6f6bffb968abf17770b4062ddbb3eef65adbfe46ee8be32951b3eb1208dc09c3c70be0783923ecbd3fabfae7748bfa37bf13ff617c53f5967c43f8d23cb3f9d1809401740cfbe6e8d3ec0e8811fc007292040681390c0c7cbf33fed0cc340aa716840894043c011398dc0cb1ec13ee593444084ce99bff2f248bffbbf44c04c4d803f80422540b5728abf930c073f1ff379be80ef6fbcf038bebfc29988bf6f4f9b3fb71c1fc02856a7bfc64c8cbff9a168bf6df27b3fdb830440c8cdbe3ff1c214c030ef723f7a890cc093f780bfcf8e1b40bbeb53c0b9c3e5bdd1a0b1bda02f16405d7b69c0768396bf8898a8bf74d3aebfee3da23fca9ed6bf2d21b7bf6b15fcbf5f1c21c0df499d3fb7e4e73f0104713f81aaf5bff4ac81bf94312140020796409817e43ee1ab0a40f63cf2bfb63d253f63a790be6302f6bf3398b93e7851e0be30e01dbf50fe2540a1b5cbbff23f5ebfd030d03e6acdfabfb15291bec28cdf3f98e93fbe3954513fa220133f82a43dc08d3315405d2d3540e6ec973f71210ac079b6f63d8e9f93bf0661b4bf0d9816bf997429bfe875394023b0933ee18a01400dabc43ffc455e400422383e0a4f6dc0853eac3df95daa3fcda76ac0d6631fc0a3fa973fc8805dbf4878c73e5a4fe43e9a8047bfd669213f97228e3fb047203f530bb7bfa6783c3fe2d99040b2fbbfbfa1da8ac04792c23f141a3840293cc6bf85276b3e18972240324740c00da1b43fce802bc02c9b833f3a488fbf3ebaa9be25e25fc06084ad3fa364cd3f18136e3e41cdc43ea67228c0e0c584c0ab9ce8be0924603f28df56bf7e888cbfbca41bbe38e518bd4941cd3e90dc6cc0d6d7b8be555fc63ff3b2e73f897e3abee9773740ad7e7140ae5cd8be5071b33f83d2153f67eb4a40ba9db13f4acb9dc09a3cc4bf30ca2240e6a152c04bd662c03a7e524080302ebfc7fa4740554c773fc90b84c0820630c0aa3b8e3f7b097bbf1e0da83f89e3bb3ff8d68b3f24a542bf150ed5bfdd994ac001771fc0d3c1bebf85b35f3fbcad74c08bede4bf882194400c6e9a3f5c5aedbe3a3a18bf7d262abff17d1c40d25d643e631a93bfc8d95ac0c1e37dbf2be659bf3e8f0e3f5f9204c07d4805c04676b2bea8461f3f013638c0086781400b0d713fe9b1a3bfb862d3bf5ea4404068e0b4be21aa043ea1910c404e01a0c0b6ce61be74175cbd94c9f43ff8bad6bf37a254bf1df283c0633267c00974f7bf18a435c07420b33e8285f83f028da0bfb1fba2bf780051c0cfd20a406d7d0640797e1fbfbbe6babf7f5d73befea676bf978f37c053d03dbdd685a4bfb223c93f77ff26c0dc3c96bfa1da81bf119f0440bf59dc3eaa9db5bd4b3489be9f9bb43e27983abfedf2843e7294b44030534ec0ca0795be34ec52408a7ba4bff993433f88a2b03f8d99dd3e78e5febe7d0c84beba3b20bf4b3545be9a241ec020da254031473f40221ccfbe5e458d3e927481be47be26bf80e16a40499c28c04c66c6bdaa98c4be43ccb3be067eb23d062418bfe9c32abf7c1c6b3f96c203409d96813e0b1cc1bf86cbdfbda88e3ebf21c32ac07cd5703fd40cd13f81ed43c094f2aebf7a0c943fe1747d3ff1322f404d0c42c0fd86e0bdee98f7bfe376d9bf05772c3e7e2b0abf55788cbfa32ac43f89d5723e415f8e3fef5e513fd7f9ed3eed69a6beb3dc4a4000bc0d4048a8cb3e397a25bed4e4d93fb7c69dbf96abbebf9fd20f3fc8ae24c0af641d40d7def6bf338e08409bce0bbf57333f3e41f516c03d9fd5bf41863e3e54e5ad3f7b2815c07db243bf4a2188405059993d10f5d2bf525f72405ceacf3f8d5bd4beca4035c002e405be37fb9f3f54693940b924833f1f90c9bf6d457b3ee6cb70c05fc8c0bfb2d61bbfcced8a3f906bf3be5f62823f8c7a154007ea7fc0b616423f8c04603fae3a66bffe8d3c408f13abbfbe4251bfbf10e33f60a756c0"), dtype=np.float32).reshape([1, 3, 32, 32])


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
